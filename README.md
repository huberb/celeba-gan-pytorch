classic GAN on CelebA dataset

example output after one epoch:

![Example Image](/output/3000.png?raw=true "Example Image")

generator model:
```
----------------------------------------------------------------                                                                                                                               
        Layer (type)               Output Shape         Param #                                                                                                                                
================================================================                                                                                                                               
   ConvTranspose2d-1           [-1, 1024, 4, 4]       2,098,176                                                                                   
       BatchNorm2d-2           [-1, 1024, 4, 4]           2,048                                                                                   
              ReLU-3           [-1, 1024, 4, 4]               0                                                                                   
   ConvTranspose2d-4            [-1, 512, 8, 8]       8,389,120                                                                                   
       BatchNorm2d-5            [-1, 512, 8, 8]           1,024                                                                                   
              ReLU-6            [-1, 512, 8, 8]               0                                                                                   
   ConvTranspose2d-7          [-1, 256, 16, 16]       2,097,408                                                                                   
       BatchNorm2d-8          [-1, 256, 16, 16]             512                                                                                   
              ReLU-9          [-1, 256, 16, 16]               0                                                                                   
  ConvTranspose2d-10          [-1, 128, 32, 32]         524,416                                                                                   
      BatchNorm2d-11          [-1, 128, 32, 32]             256                                                                                   
             ReLU-12          [-1, 128, 32, 32]               0                                                                                   
  ConvTranspose2d-13           [-1, 64, 64, 64]         131,136                                                                                   
      BatchNorm2d-14           [-1, 64, 64, 64]             128                                                                                   
             ReLU-15           [-1, 64, 64, 64]               0                                                                                   
  ConvTranspose2d-16          [-1, 3, 128, 128]           3,075                                                                                   
             Tanh-17          [-1, 3, 128, 128]               0                                                                                   
================================================================                                                                                  
Total params: 13,247,299                                                                                                                          
Trainable params: 13,247,299                                                                                                                      
Non-trainable params: 0                                                                                                                           
----------------------------------------------------------------                                                                                  
Input size (MB): 0.00                                                                                                                             
Forward/backward pass size (MB): 12.38                                                                                                            
Params size (MB): 50.53                                                                                                                           
Estimated Total Size (MB): 62.91
```

discriminator model:
```
----------------------------------------------------------------                                                                                  
        Layer (type)               Output Shape         Param #                                                                                   
================================================================                                                                                  
            Conv2d-1           [-1, 64, 64, 64]           3,136                                                                                   
         LeakyReLU-2           [-1, 64, 64, 64]               0                                                                                   
            Conv2d-3          [-1, 128, 32, 32]         131,200                                                                                   
       BatchNorm2d-4          [-1, 128, 32, 32]             256                                                                                   
         LeakyReLU-5          [-1, 128, 32, 32]               0                                                                                   
            Conv2d-6          [-1, 256, 16, 16]         524,544
       BatchNorm2d-7          [-1, 256, 16, 16]             512 
         LeakyReLU-8          [-1, 256, 16, 16]               0                                
            Conv2d-9            [-1, 512, 8, 8]       2,097,664                                
      BatchNorm2d-10            [-1, 512, 8, 8]           1,024                                
        LeakyReLU-11            [-1, 512, 8, 8]               0 
           Conv2d-12           [-1, 1024, 4, 4]       8,389,632                                
      BatchNorm2d-13           [-1, 1024, 4, 4]           2,048
        LeakyReLU-14           [-1, 1024, 4, 4]               0                                
           Conv2d-15              [-1, 1, 1, 1]          16,385                                
          Sigmoid-16              [-1, 1, 1, 1]               0                                
================================================================                               
Total params: 11,166,401                       
Trainable params: 11,166,401                   
Non-trainable params: 0                        
----------------------------------------------------------------                               
Input size (MB): 0.19                          
Forward/backward pass size (MB): 9.63                                                          
Params size (MB): 42.60                        
Estimated Total Size (MB): 52.41               
----------------------------------------------------------------
```
