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

    def __init__(self,mask,input_dim, z_dim_gs, z_dim_uns ,hidden_dims_dec, hidden_dims_enc_gs, hidden_dims_enc_uns, use_cuda = False, n_gs=30, batch_size = 128, num_iafs = 5, iaf_dim = 40):
        super().__init__()
        
        self.idx_gs = mask.sum(dim=0) > 0
        mask = mask[:,self.idx_gs]
        # create the encoder and decoder networks

        if z_dim_gs > 0:
            self.encoder_gs = Encoder_GSEA(mask.shape[1], z_dim_gs, hidden_dims_enc_gs, n_gs, mask, batch_size)
        if z_dim_uns > 0:
            self.encoder_uns = Encoder(input_dim, z_dim_uns, hidden_dims_enc_uns)
        
        self.decoder = Decoder(input_dim, z_dim_gs + z_dim_uns, hidden_dims_dec)
        
        self.mask = mask
       
        
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()


        self.use_cuda = use_cuda
        
        self.z_dim_gs = z_dim_gs 
        self.z_dim_uns = z_dim_uns
        self.z_dim = z_dim_gs + z_dim_uns
        
        self.iafs = [ affine_autoregressive(z_dim_uns, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)
        
        self.num_iafs = num_iafs
        
        self.iafs2 = [ affine_autoregressive(z_dim_gs, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules2 = nn.ModuleList(self.iafs2)


    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        pyro.module("iafs_modules", self.iafs_modules)
        pyro.module("iafs_modules2", self.iafs_modules2)
        

        # sample from prior (value will be sampled by guide when computing the ELBO)
        with pyro.plate("data", x.shape[1]):
            
            
            # setup hyperparameters for prior p(z)
            if self.z_dim_gs > 0 :
                z_loc_gs = x.new_zeros(torch.Size((x.shape[1], self.z_dim_gs))) 
                z_scale_gs = x.new_ones(torch.Size((x.shape[1], self.z_dim_gs))) 

                z_loc_gs = pyro.sample("z_loc_gs", dist.Delta(z_loc_gs, event_dim=1))
                z_scale_gs = pyro.sample("z_scale_gs", dist.Delta(z_scale_gs, event_dim=1))
        
            # setup hyperparameters for prior p(z)
            if self.z_dim_uns > 0 :
                z_loc_uns = x.new_zeros(torch.Size((x.shape[1], self.z_dim_uns)))
                z_scale_uns = x.new_ones(torch.Size((x.shape[1], self.z_dim_uns))) 

                z_loc_uns = pyro.sample("z_loc_uns", dist.Delta(z_loc_uns, event_dim=1))
                z_scale_uns = pyro.sample("z_scale_uns", dist.Delta(z_scale_uns, event_dim=1))
        
            #z1 = pyro.sample("latent_gs", dist.Normal(z_loc_gs, z_scale_gs).to_event(1))
            if self.num_iafs > 0:
                if self.z_dim_gs > 0 : 
                    z1 = pyro.sample("latent_gs",dist.TransformedDistribution(dist.Normal(z_loc_gs, z_scale_gs), self.iafs2))
                if self.z_dim_uns > 0 : 
                    z2 = pyro.sample("latent_uns",dist.TransformedDistribution(dist.Normal(z_loc_uns, z_scale_uns), self.iafs))
            else:
                if self.z_dim_gs > 0 : 
                    z1 = pyro.sample("latent_gs",dist.Normal(z_loc_gs, z_scale_gs).to_event(1) )
                if self.z_dim_uns > 0 : 
                    z2 = pyro.sample("latent_uns",dist.Normal(z_loc_uns, z_scale_uns).to_event(1) )

            #print(z2.shape)
            
            # decode the latent code z
            if self.z_dim_gs > 0 and self.z_dim_uns > 0 :
                loc_gene, scale_gene = self.decoder(torch.cat((z1,z2), dim = 1))
            if self.z_dim_gs > 0 and self.z_dim_uns == 0:
                loc_gene, scale_gene = self.decoder(z1)
            if self.z_dim_gs == 0 and self.z_dim_uns > 0:
                loc_gene, scale_gene = self.decoder(z2)
                      
            lk = dist.Normal(loc_gene.t(), scale_gene.t()).log_prob(x[0,:,:].t().squeeze())

            
            #penalty = kl_div(z1,z2, reduction = "batchmean")
            if self.z_dim_uns > 0 and self.z_dim_gs > 0:
                penalty = self.kl_regularization(z_loc_gs, z_loc_uns, z_scale_gs, z_scale_uns)
            else:
                penalty = 0
            

            # using mean instead of sum would make the loss not sensitive to the batch size

            pyro.factor("loss", lk.sum() + penalty)
         

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder_gs` and `encoder_uns` with Pyro

        if self.z_dim_gs > 0:
            pyro.module("encoder_gs", self.encoder_gs)
        if self.z_dim_uns > 0:
            pyro.module("encoder_uns", self.encoder_uns)
        
        # sample the latent code z
        with pyro.plate("data", x.shape[1]):
            
            # use the encoder to get the parameters used to define q(z|x)
            
            if self.z_dim_gs > 0:
                z_loc_gs, z_scale_gs = self.encoder_gs(x[:,:,self.idx_gs])

                z_loc_gs = pyro.sample("z_loc_gs", dist.Delta(z_loc_gs, event_dim=1))
                z_scale_gs = pyro.sample("z_scale_gs", dist.Delta(z_scale_gs, event_dim=1))

            if self.z_dim_uns > 0 :
                z_loc_uns, z_scale_uns = self.encoder_uns(x[0,:,:].squeeze())

                z_loc_uns = pyro.sample("z_loc_uns", dist.Delta(z_loc_uns, event_dim=1))
                z_scale_uns = pyro.sample("z_scale_uns", dist.Delta(z_scale_uns, event_dim=1))

            
            if self.num_iafs > 0:
                if self.z_dim_gs > 0 : 
                    z1 = pyro.sample("latent_gs",dist.TransformedDistribution(dist.Normal(z_loc_gs, z_scale_gs), self.iafs2))
                if self.z_dim_uns > 0 : 
                    z2 = pyro.sample("latent_uns",dist.TransformedDistribution(dist.Normal(z_loc_uns, z_scale_uns), self.iafs))
            else:
                if self.z_dim_gs > 0 : 
                    z1 = pyro.sample("latent_gs",dist.Normal(z_loc_gs, z_scale_gs).to_event(1) )
                if self.z_dim_uns > 0 : 
                    z2 = pyro.sample("latent_uns",dist.Normal(z_loc_uns, z_scale_uns).to_event(1) )


    # define a helper function for enforcing separation
    def kl_regularization(self, z_loc_gs, z_loc_uns, z_scale_gs, z_scale_uns):
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # Beta variational autoencoder could be an option

        # try using torch.mean() instead of torch.sum()

        # Do we need the mean of the z_scale_gs and z_loc_gs to compare with z_scale_uns and z_loc_uns?

        #z_loc_gs_mean = torch.mean(z_loc_gs,1)
        #z_scale_gs_mean = torch.mean(z_scale_gs,1)
        #z_loc_uns_mean = torch.mean(z_loc_uns,1)
        #z_scale_uns_mean = torch.mean(z_scale_uns,1)

        
        #pen1 = (  torch.log(z_scale_gs_mean) - torch.log(z_scale_uns_mean)  )  +  (  (z_scale_uns_mean**2) + (z_loc_uns_mean - z_loc_gs_mean)**2  ) / (  2 * z_scale_gs_mean**2  )   -   0.5 
        #pen2 = (  torch.log(z_scale_uns_mean) - torch.log(z_scale_gs_mean)  )  +  (  (z_scale_gs_mean**2) + (z_loc_gs_mean - z_loc_uns_mean)**2  ) / (  2 * z_scale_uns_mean**2  )   -   0.5
        
        #pen = torch.mean(pen1 + pen2)

        #pen = torch.mean( (  torch.log(z_scale_gs_mean) - torch.log(z_scale_uns_mean)  )  +  (  (z_scale_uns_mean**2) + (z_loc_uns_mean - z_loc_gs_mean)**2  ) / (  2 * z_scale_gs_mean**2  )   -   0.5 )

        #pen = torch.sum( (  torch.log(z_scale_gs) - torch.log(z_scale_uns)  )  +  (  (z_scale_uns**2) + (z_loc_uns - z_loc_gs)**2  ) / (  2 * z_scale_gs**2  )   -   0.5 )

        #pen = torch.sum(  torch.log(z_scale_gs) + torch.log(z_scale_uns)  +  0.5 * (  (z_scale_uns**2) + (z_loc_uns)**2  )  +   0.5 * (  (z_scale_gs**2) + (z_loc_gs)**2  )  -   1 )

        totpen = []

        for i in range(len(z_loc_gs[1,:])):
            for j in range(len(z_loc_uns[1,:])):
                kl1 = torch.mean((  torch.log(z_scale_gs[:,i]) - torch.log(z_scale_uns[:,j])  )  +  (  (z_scale_uns[:,j]**2) + (z_loc_uns[:,j] - z_loc_gs[:,i])**2  ) / (  2 * z_scale_gs[:,i]**2  )   -   0.5)
                kl2 = torch.mean((  torch.log(z_scale_uns[:,j]) - torch.log(z_scale_gs[:,i])  )  +  (  (z_scale_gs[:,i]**2) + (z_loc_gs[:,i] - z_loc_uns[:,j])**2  ) / (  2 * z_scale_uns[:,j]**2  )   -   0.5)
                totpen.append((kl1+kl2)/2)

        pen = torch.mean(torch.as_tensor(totpen))

        return pen


    # define a helper function for reconstructing cells
    def reconstruct_cell(self, x):
        
        z = self.sample_hidden_dims(x, sample = False)
        cell_values, loc = self.decoder(z)
        
        return cell_values
   

    # define a helper function for reconstructing cells
    def sample_hidden_dims(self, x, sample = True, concat = True):

        if self.z_dim_gs > 0:
            z_loc_gs, z_scale_gs = self.encoder_gs(x[:,:,self.idx_gs])
        if self.z_dim_uns > 0:
            z_loc_uns, z_scale_uns = self.encoder_uns(x[0,:,:].squeeze())

        if sample:
            # sample in latent space
            if self.z_dim_gs > 0 :
                if self.num_iafs > 0:
                    z1 = dist.TransformedDistribution(dist.Normal(z_loc_gs, z_scale_gs), self.iafs2).sample()
                else:
                    z1 = dist.Normal(z_loc_gs, z_scale_gs).sample()
            if self.z_dim_uns > 0 :
                if self.num_iafs > 0:
                    z1 = dist.TransformedDistribution(dist.Normal(z_loc_uns, z_scale_uns), self.iafs2).sample()
                else:
                    z1 = dist.Normal(z_loc_uns, z_scale_uns).sample()
                   
        else:
            #return MAP
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
            if self.z_dim_gs > 0 and self.z_dim_uns > 0 :
                return torch.cat((z1,z2), dim = 1)
            if self.z_dim_gs > 0 and self.z_dim_uns == 0:
                return z1
            if self.z_dim_gs == 0 and self.z_dim_uns > 0:
                return z2          
            
        else:
            if self.z_dim_gs > 0 and self.z_dim_uns > 0 :
                return (z1,z2)
            if self.z_dim_gs > 0 and self.z_dim_uns == 0:
                return z1
            if self.z_dim_gs == 0 and self.z_dim_uns > 0:
                return z2  



    
    