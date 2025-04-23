import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functorch import jacfwd, vmap
import math

def jac(f, z):
    # composed with vmap for batched Jacobians
    return vmap(jacfwd(f))(z).squeeze(1)
    
def jac_robust(f, z, create_graph=False):
    # alternative jac if experiencing crashes 
    batch_size, z_dim = z.size()
    v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(z)
    z = z.repeat(1, z_dim).view(-1, z_dim)
    return torch.autograd.functional.jvp(f, z, v=v, create_graph=create_graph)[1].view(batch_size, z_dim, -1).permute(0, 2, 1)

# ---------------------------
# - Variational Autoencoder -
# ---------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2) # mu + log_var
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            torch.nn.Sigmoid())
        
    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, library_size=None):
        x_hat = self.decoder(z)
        if library_size is not None:
            x_hat = x_hat * library_size
        return x_hat
    
    def forward(self, x, library_size=None):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, library_size)
        return x_hat, mu, log_var

    def loss(self, x, x_hat, mu, log_var):
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_div

    # sample around cell
    def sample(self, x, n=10, scale=1., library_size=None):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = scale * torch.randn(n, self.latent_dim)
        z = mu + eps * std
        x_hat = self.decode(z, library_size)
        return x_hat, z
    
    def make_step(self, z, i, delta=0.05, library_size=None):
        '''
        Make a step in the latent space, following the direction 
        of the ith row of the decoder's Jacobian matrix.
        '''
        J = jac(lambda x: self.decode(x, library_size)[:, i], z)

        # if i contains several genes
        if isinstance(i, list) or isinstance(i, np.ndarray) or isinstance(i, torch.Tensor):
            J = J.mean(axis=1)

        z = z - delta * J[:,i,:] # subtract jacobian row corresponding to chosen gene i
        return z

    def grad_wrt_i(self, z, i, library_size=None):
        '''
        Compute the gradient of the decoder's output w.r.t. the ith gene.
        '''
        J = jac(lambda x : self.decode(x, library_size), z)
        return J[:,i,:]


# ---------------------------
# -  Negative Binomial VAE  -
# ---------------------------
class NBVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, r_init=2, scaling_type="library", extra_outputs=0):
        super(NBVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.extra_outputs = extra_outputs
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim * 2) # mu + log_var
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, input_dim + extra_outputs),)
        
        self.nb = NBLayer(input_dim, r_init=r_init, scaling_type=scaling_type)
        
    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        z = z.view(-1, self.latent_dim)
        x_hat = self.decoder(z)

        if self.extra_outputs:
            gene_part, extra_part = x_hat.split([self.input_dim, self.extra_outputs], dim=1)
            x_hat = torch.cat((self.nb(gene_part), extra_part), dim=1)
        else:
            x_hat = self.nb(x_hat)
            
        return x_hat
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def loss(self, x, x_hat, mu, log_var, scaling):
        recon_loss = self.nb.loss(x, scaling, x_hat)
        recon_loss = recon_loss.sum()
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_div

    # sample around cell
    def sample(self, x, n=10, scale=1.):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = scale * torch.randn(n, self.latent_dim)
        z = mu + eps * std
        x_hat = self.decode(z)
        return x_hat, z
    
    def make_step(self, z, i, delta=0.05):
        '''
        Make a step in the latent space, following the direction 
        of the ith row of the decoder's Jacobian matrix.
        '''
        J = jac(lambda x: self.decode(x)[:, i], z)

        # if i contains several genes
        if isinstance(i, list) or isinstance(i, np.ndarray) or isinstance(i, torch.Tensor):
            J = J.mean(axis=1)

        z = z - delta * J 
        return z

    def grad_wrt_i(self, z, i):
        '''
        Compute the gradient of the decoder's output w.r.t. the ith gene.
        '''
        J = jac(self.decode, z).squeeze()
        return J[:,i,:]

class NBLayer(nn.Module):
    '''
    Schuster and Krogh (2023)
    A negative binomial of scaled values of m and learned parameters for r.
    mhat = m/M, where M is the scaling factor

    The scaled value mhat is typically the output value from the NN

    If rhat=None, it is assumed that the last half of mhat contains rhat.

    m = M*mhat
    '''
    def __init__(self, out_dim, r_init, scaling_type='library',reduction='none'):
        super(NBLayer, self).__init__()

        # initialize parameter for r
        # real-valued positive parameters are usually used as their log equivalent
        self.log_r = torch.nn.Parameter(torch.full(fill_value=math.log(r_init), size=(1,out_dim)), requires_grad=True)
        #self.log_r = torch.nn.Parameter(torch.full(fill_value=math.log(r_init), size=(1,out_dim)), requires_grad=False)
        # determine the type of activation based on scaling type
        if scaling_type in ['library','total_count']:
            self.activation = 'sigmoid'
        elif scaling_type in ['mean','median']:
            self.activation = 'softplus'
        else:
            raise ValueError('Unknown scaling type specified. Please use one of: "library", "total_count", "mean", or "median".')
        self.reduction = reduction
    
    def forward(self, x):
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            #return F.relu(x)
            return F.softplus(x)

    # Convert to m from scaled variables
    def rescale(self,M,mhat):
        return M*mhat

    def loss(self,x,M,mhat):
        if self.reduction == 'none':
            return -logNBdensity(x,self.rescale(M,mhat),torch.exp(self.log_r))
        elif self.reduction == 'sum':
            return -logNBdensity(x,self.rescale(M,mhat),torch.exp(self.log_r)).sum()

    # The logprob of the tensor
    def logprob(self,x,M,mhat):
        return logNBdensity(x,self.rescale(M,mhat),torch.exp(self.log_r))

    def sample(self,nsample,M,mhat):
        # Note that torch.distributions.NegativeBinomial returns FLOAT and not int
        with torch.no_grad():
            # m = pr/(1-p), so p = m/(m+r)
            m = self.rescale(M,mhat)
            probs = m/(m+torch.exp(self.log_r))
            nb = torch.distributions.NegativeBinomial(torch.exp(self.log_r), probs=probs)
            return nb.sample([nsample]).squeeze()

def logNBdensity(k,m,r):
  ''' 
  Negative Binomial NB(k;m,r), where m is the mean and r is "number of failures"
  r can be real number (and so can k)
  k, and m are tensors of same shape
  r is tensor of shape (1, n_genes)
  Returns the log NB in same shape as k
  '''
  # remember that gamma(n+1)=n!
  eps = 1.e-10
  c = 1./(r+m+eps)
  # Under-/over-flow protection is added
  x = torch.lgamma(k+r) - torch.lgamma(r) - torch.lgamma(k+1) + k*torch.log(m*c+eps) + r*torch.log(r*c)
  return x