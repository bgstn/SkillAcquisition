# Skill Learning with clustering
This repo tries to learn to segment skills from image demonstrations and learn a controller for each skill. Each controller has a goal and gain as its own private property. 

# Modelling process

$$
\begin{aligned}
    z_t &= Encoder(Im_t) \\
    \delta_t &= Switching(Im_t) \\
    k_t &= lookup(K, \delta_t) \\
    g_t &= (G, \delta_t) \\
    u_t &= k_t * (g_t - z_t) \\
    Im_t &= Decoder(z_t)
\end{aligned}
$$

where Encoder Decoder and Switching are separate neural networks. 

# Job

The following code starts a job using a toy dataset.
```
python -u train.py -m "test"
```
