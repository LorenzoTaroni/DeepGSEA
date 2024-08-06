import pyro
import torch
import torch.nn as nn
import pyro.distributions as dist
from svgsa.Decoder import Decoder
from svgsa.Encoder_GSEA import Encoder_GSEA
from svgsa.Encoder import Encoder
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from torch.nn.functional import kl_div

class VAE(nn.Module):
    def __init__(self, mask, input_dim, z_dim_gs, z_dim_uns, hidden_dims_dec, hidden_dims_enc_gs, hidden_dims_enc_uns, beta, use_cuda=False, n_gs=30, batch_size=128, num_iafs=5, iaf_dim=40):
        super().__init__()
        
        # Filter the mask to retain columns with any non-zero values
        self.idx_gs = mask.sum(dim=0) > 0
        mask = mask[:, self.idx_gs]

        # Initialize encoders and decoder based on specified dimensions
        if z_dim_gs > 0:
            self.encoder_gs = Encoder_GSEA(mask.shape[1], z_dim_gs, hidden_dims_enc_gs, n_gs, mask, batch_size)
        if z_dim_uns > 0:
            self.encoder_uns = Encoder(input_dim, z_dim_uns, hidden_dims_enc_uns)
        
        self.decoder = Decoder(input_dim, z_dim_gs + z_dim_uns, hidden_dims_dec)
        
        # Set class attributes
        self.mask = mask
        self.beta = beta
        self.use_cuda = use_cuda
        self.z_dim_gs = z_dim_gs 
        self.z_dim_uns = z_dim_uns
        self.z_dim = z_dim_gs + z_dim_uns
        
        # Initialize IAFs (Inverse Autoregressive Flows) for variational inference
        self.iafs = [affine_autoregressive(z_dim_uns, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)
        self.num_iafs = num_iafs
        
        self.iafs2 = [affine_autoregressive(z_dim_gs, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules2 = nn.ModuleList(self.iafs2)

        # Move model to GPU if CUDA is enabled
        if use_cuda:
            self.cuda()

    def model(self, x):
        # Register the decoder with Pyro
        pyro.module("decoder", self.decoder)

        # Register IAF modules with Pyro if they are defined
        if self.num_iafs > 0:
            pyro.module("iafs_modules", self.iafs_modules)
            pyro.module("iafs_modules2", self.iafs_modules2)
        
        # Define the prior distributions for the latent variables
        with pyro.plate("data", x.shape[1]):
            if self.z_dim_gs > 0:
                z_loc_gs_init = x.new_zeros(torch.Size((x.shape[1], self.z_dim_gs))) 
                z_scale_gs_init = x.new_ones(torch.Size((x.shape[1], self.z_dim_gs))) 
                z_loc_gs_prior = pyro.sample("z_loc_gs", dist.Normal(z_loc_gs_init, x.new_ones(torch.Size((x.shape[1], self.z_dim_gs)))).to_event(1))
                z_scale_gs_prior = pyro.sample("z_scale_gs", dist.InverseGamma(z_scale_gs_init, 2 * x.new_ones(torch.Size((x.shape[1], self.z_dim_gs)))).to_event(1))
        
            if self.z_dim_uns > 0:
                z_loc_uns_init = x.new_zeros(torch.Size((x.shape[1], self.z_dim_uns)))
                z_scale_uns_init = x.new_ones(torch.Size((x.shape[1], self.z_dim_uns)))
                z_loc_uns_prior = pyro.sample("z_loc_uns", dist.Normal(z_loc_uns_init, x.new_ones(torch.Size((x.shape[1], self.z_dim_uns)))).to_event(1))
                z_scale_uns_prior = pyro.sample("z_scale_uns", dist.InverseGamma(z_scale_uns_init, 2 * x.new_ones(torch.Size((x.shape[1], self.z_dim_uns)))).to_event(1))

            # Sample latent variables using IAFs if they are defined
            if self.num_iafs > 0:
                if self.z_dim_gs > 0:
                    z1 = pyro.sample("latent_gs", dist.TransformedDistribution(dist.Normal(z_loc_gs_prior, z_scale_gs_prior), self.iafs2))
                if self.z_dim_uns > 0:
                    z2 = pyro.sample("latent_uns", dist.TransformedDistribution(dist.Normal(z_loc_uns_prior, z_scale_uns_prior), self.iafs))
            else:
                if self.z_dim_gs > 0:
                    z1 = pyro.sample("latent_gs", dist.Normal(z_loc_gs_prior, z_scale_gs_prior).to_event(1))
                if self.z_dim_uns > 0:
                    z2 = pyro.sample("latent_uns", dist.Normal(z_loc_uns_prior, z_scale_uns_prior).to_event(1))

            # Decode the latent variables to reconstruct the input
            if self.z_dim_gs > 0 and self.z_dim_uns > 0:
                loc_gene, scale_gene = self.decoder(torch.cat((z1, z2), dim=1))
            elif self.z_dim_gs > 0:
                loc_gene, scale_gene = self.decoder(z1)
            else:
                loc_gene, scale_gene = self.decoder(z2)
            
            # Calculate log probability of the input given the reconstruction
            lk = dist.Normal(loc_gene.t(), scale_gene.t()).log_prob(x[0, :, :].t().squeeze())

            # Compute KL penalty if both gs and uns dimensions are used
            if self.z_dim_uns > 0 and self.z_dim_gs > 0:
                penalty = self.kl_regularization(z_loc_gs_prior, z_loc_uns_prior, z_scale_gs_prior, z_scale_uns_prior)
            else:
                penalty = torch.tensor([0])
            
            # Define the loss function with optional KL regularization
            if self.z_dim_uns > 0 and self.z_dim_gs > 0:
                pyro.factor("loss", torch.sum(lk) - self.beta * torch.sum(1 / penalty))
            else:
                pyro.factor("loss", torch.sum(lk))

    # Define the guide q(z|x)
    def guide(self, x):
        # Register encoders with Pyro
        if self.z_dim_gs > 0:
            pyro.module("encoder_gs", self.encoder_gs)
        if self.z_dim_uns > 0:
            pyro.module("encoder_uns", self.encoder_uns)
        
        # Define the variational distributions for the latent variables
        with pyro.plate("data", x.shape[1]):
            if self.z_dim_gs > 0:
                z_loc_gs, z_scale_gs = self.encoder_gs(x[:, :, self.idx_gs])
                z_loc_gs_prior = pyro.sample("z_loc_gs", dist.Delta(z_loc_gs, event_dim=1))
                z_scale_gs_prior = pyro.sample("z_scale_gs", dist.Delta(z_scale_gs, event_dim=1))

            if self.z_dim_uns > 0:
                z_loc_uns, z_scale_uns = self.encoder_uns(x[0, :, :].squeeze())
                z_loc_uns_prior = pyro.sample("z_loc_uns", dist.Delta(z_loc_uns, event_dim=1))
                z_scale_uns_prior = pyro.sample("z_scale_uns", dist.Delta(z_scale_uns, event_dim=1))

            # Sample latent variables using IAFs if they are defined
            if self.num_iafs > 0:
                if self.z_dim_gs > 0:
                    z1 = pyro.sample("latent_gs", dist.TransformedDistribution(dist.Normal(z_loc_gs_prior, z_scale_gs_prior), self.iafs2))
                if self.z_dim_uns > 0:
                    z2 = pyro.sample("latent_uns", dist.TransformedDistribution(dist.Normal(z_loc_uns_prior, z_scale_uns_prior), self.iafs))
            else:
                if self.z_dim_gs > 0:
                    z1 = pyro.sample("latent_gs", dist.Normal(z_loc_gs_prior, z_scale_gs_prior).to_event(1))
                if self.z_dim_uns > 0:
                    z2 = pyro.sample("latent_uns", dist.Normal(z_loc_uns_prior, z_scale_uns_prior).to_event(1))
                    
    # Compute KL divergence regularization term for each pair of latent dimensions
    def kl_regularization(self, z_loc_gs, z_loc_uns, z_scale_gs, z_scale_uns):
       
        totpen = []

        for i in range(len(z_loc_gs[1, :])):
            for j in range(len(z_loc_uns[1, :])):
                kl1 = (torch.log(z_scale_gs[:, i]) - torch.log(z_scale_uns[:, j])) + \
                      ((z_scale_uns[:, j] ** 2) + (z_loc_uns[:, j] - z_loc_gs[:, i]) ** 2) / (2 * z_scale_gs[:, i] ** 2) - 0.5
                kl2 = (torch.log(z_scale_uns[:, j]) - torch.log(z_scale_gs[:, i])) + \
                      ((z_scale_gs[:, i] ** 2) + (z_loc_gs[:, i] - z_loc_uns[:, j]) ** 2) / (2 * z_scale_uns[:, j] ** 2) - 0.5
                totpen.append((kl1 + kl2) / 2)

        totpen = torch.stack(totpen) 
        pen = torch.mean(totpen, dim=0)

        return pen

    # Define a helper function for reconstructing cells
    def reconstruct_cell(self, x):
        z = self.sample_hidden_dims(x, sample=False)
        cell_values, loc = self.decoder(z)
        
        return cell_values
    
    # Define a helper function to access the enrichment score layer
    def return_gs(self, x, return_geneset=False):
        score = self.encoder_gs(x[:, :, self.idx_gs], return_geneset)
        print("score: ", score)
        return score

    # Sample hidden dimensions or return MAP estimates
    def sample_hidden_dims(self, x, sample=True, concat=True):
        if self.z_dim_gs > 0:
            z_loc_gs, z_scale_gs = self.encoder_gs(x[:, :, self.idx_gs])
        if self.z_dim_uns > 0:
            z_loc_uns, z_scale_uns = self.encoder_uns(x[0, :, :].squeeze())

        if sample:
            if self.z_dim_gs > 0:
                if self.num_iafs > 0:
                    z1 = dist.TransformedDistribution(dist.Normal(z_loc_gs, z_scale_gs), self.iafs2).sample()
                else:
                    z1 = dist.Normal(z_loc_gs, z_scale_gs).sample()
            if self.z_dim_uns > 0:
                if self.num_iafs > 0:
                    z1 = dist.TransformedDistribution(dist.Normal(z_loc_uns, z_scale_uns), self.iafs2).sample()
                else:
                    z1 = dist.Normal(z_loc_uns, z_scale_uns).sample()
        else:
            if self.z_dim_gs > 0:
                z1 = z_loc_gs
            if self.num_iafs > 0:
                for i in range(len(self.iafs_modules2)):
                    z1 = self.iafs_modules2[i](z1)
                
            if self.z_dim_uns > 0:
                z2 = z_loc_uns
            if self.num_iafs > 0:
                for i in range(len(self.iafs_modules)):
                    z2 = self.iafs_modules[i](z2)
                
        if concat:
            if self.z_dim_gs > 0 and self.z_dim_uns > 0:
                return torch.cat((z1, z2), dim=1)
            if self.z_dim_gs > 0 and self.z_dim_uns == 0:
                return z1
            if self.z_dim_gs == 0 and self.z_dim_uns > 0:
                return z2
        else:
            if self.z_dim_gs > 0 and self.z_dim_uns > 0:
                return (z1, z2)
            if self.z_dim_gs > 0 and self.z_dim_uns == 0:
                return z1
            if self.z_dim_gs == 0 and self.z_dim_uns > 0:
                return z2  



    
    