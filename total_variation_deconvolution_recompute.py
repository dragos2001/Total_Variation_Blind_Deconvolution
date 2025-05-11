import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import cg
from scipy.ndimage import convolve
from blind_deconvolution import rotated_anisotropic_gaussian_kernel , blur_kernel
from math import pi,e
from numpy.fft import fft2, ifft2, fftshift
import cv2
import matplotlib.pyplot as plt
from L1_support import add_gaussian_noise , calculate_ssim , peak_signal_noise_ratio
from blind_deconvolution import fft_deconvolution
from scipy.sparse.linalg import LinearOperator

def gradient(u):
    ux = np.roll(u, -1, axis=1) - u
    uy = np.roll(u, -1, axis=0) - u
    return ux, uy

def divergence(px, py):
    pxm = px - np.roll(px, 1, axis=1)
    pym = py - np.roll(py, 1, axis=0)
    return pxm + pym

def tv_weights(u, epsilon=1e-3):
    ux, uy = gradient(u)
    return 1.0 / np.sqrt(ux**2 + uy**2 + epsilon)

def solve_u(f, h, u0, lambda1, max_fp_iter=10, rtol=1e-5,atol=0.1):
    u = u0.copy()
    h_flipped = np.flip(np.flip(h, axis=0), axis=1)

    for _ in range(max_fp_iter):
        w = tv_weights(u)

        def A(u_flat):
            u_img = u_flat.reshape(f.shape)
            Hu = convolve2d(u_img, h, mode='same', boundary='symm')
            HTHu = convolve2d(Hu, h_flipped, mode='same', boundary='symm')
            gradient_ux , gradient_uy = gradient(u)
            div_grad = divergence(w * gradient_ux , w * gradient_uy)
            
            return (HTHu - lambda1 * div_grad).flatten()
        
        A_operator = LinearOperator ( shape = (f.size, f.size) , matvec = A)
        b_img = convolve2d (f, h_flipped, mode='same', boundary='symm')
        b = b_img.flatten()

        u_flat, _ = cg(A_operator, b, x0 = u.flatten() , rtol = rtol)
        u = u_flat.reshape(f.shape)
        
        #Normalize
        u[u<0]=0
        
    return u

def solve_h(f, u, h0, lambda2, max_fp_iter=10, rtol=1e-5,atol=0.1):
    h = h0.copy()
    u_flipped = np.flip(np.flip(u, axis=0), axis=1)

    for _ in range(max_fp_iter):
        w = tv_weights(h)

        def A(h_flat):
            h_img = h_flat.reshape(h.shape)
            UH = convolve2d(h_img, u, mode='same', boundary='symm')
            UTUH = convolve2d(UH, u_flipped, mode='same', boundary='symm')
            gradient_hx , gradient_hy = gradient(h)
            div_grad = divergence(w*gradient_hx, w*gradient_hy)
            return (UTUH - lambda2 * div_grad).flatten()
        
        A_operator = LinearOperator ( shape = (h0.size, h0.size) , matvec = A)
        b_img = convolve2d(f, u_flipped, mode='same', boundary='symm')
        b = b_img.flatten()

        h_flat, _ = cg(A_operator, b, x0=h.flatten(), rtol=rtol)
        h = h_flat.reshape(h.shape)

        # Normalize and project
        h[h < 0] = 0
        h= (h + h.transpose)//2
        h /= (np.sum(h) + 1e-8)

    return h

def blind_deconvolution_am(f, kernel_shape, lambda1, lambda2, num_am_iter=10):
    u = f.copy()
    h = np.zeros(kernel_shape)
    h[kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0  # delta init

    for i in range(num_am_iter):
        print(f"AM Iteration {i+1}")
        u = solve_u(f, h, u, lambda1)
        h = solve_h(f, u, h, lambda2)

    return u, h

if __name__  ==  "__main__" :
    #kernel
    kernel_size=(5,5)
    kernel_matrix_gaussian_blur = blur_kernel(sigma = 0.5, shape = kernel_size)
    kernel_matrix_motion_blur = rotated_anisotropic_gaussian_kernel(kernel_size[0] , sigma_x = 2 , sigma_y=10, theta=pi/4)
    
    fig1,axs1 = plt.subplots(1,2)
    fig1.suptitle("Blur Kernels")
    
    #Axes 0
    axs1[ 0].set_title("Original Image")
    axs1[ 0].imshow(kernel_matrix_gaussian_blur , cmap='gray')
    axs1[ 0].axis("off")
    
    #---------------------------------
    #Axes 1
    axs1[ 1].set_title("Blurred Image")
    axs1[ 1].imshow(kernel_matrix_motion_blur , cmap='gray')
    axs1[ 1].axis("off")
    
    
    plt.show()
    
    
    
    
    
    #Read an image
    frame = cv2.imread('lena.png',cv2.IMREAD_GRAYSCALE)
    blurred_image = convolve2d(frame, kernel_matrix_gaussian_blur,mode="same")
    blurred_image = np.array( blurred_image , dtype = int )
    size=frame.shape[:2]
    
    #img
    img_motion_blur = convolve2d(frame , kernel_matrix_motion_blur,mode="same",boundary='wrap')
    
    #---------------------------------
    #Figure
    fig, axs=plt.subplots(2,2)
    
    #---------------------------------
    #Axes 0
    axs[0 , 0].set_title("Original Image")
    axs[0 , 0].imshow(frame , cmap='gray')
    axs[0 , 0].axis("off")
    
    #---------------------------------
    #Axes 1
    axs[0 , 1].set_title("Blurred Image")
    axs[0 , 1].imshow(blurred_image , cmap='gray')
    axs[0 , 1].axis("off")
    
    #---------------------------------
    #Add noise 
    noisy_blurred_image = add_gaussian_noise(blurred_image , 0 , 30)
    
    #---------------------------------
    #Axes2
    axs[1,0].set_title("Noisy Image")
    axs[1,0].imshow(noisy_blurred_image , cmap='gray')
    axs[1,0].axis("off")

    #total variation deconv-----------
    alfa1 = 2*10 **(-6)
    alfa2 = 1.5*10 **(-5)
    u_image, h_kernel = blind_deconvolution_am( noisy_blurred_image , kernel_size , alfa1 , alfa2)
   
    #Axes2
    axs[1,1].set_title("Reconstructed Image")
    axs[1,1].imshow(u_image , cmap='gray')
    axs[1,1].axis("off")

    #ssim
    ssim_original = calculate_ssim(frame , frame)
    ssim_blurred = calculate_ssim(frame , blurred_image)
    ssim_noisy_blurred = calculate_ssim(frame, noisy_blurred_image)
    ssim_rec = calculate_ssim(frame, u_image)
    
    #psnr noisy blurred
    psnr_noisy_blurred = peak_signal_noise_ratio( frame , noisy_blurred_image)
    psnr_reconstructed = peak_signal_noise_ratio( frame , u_image)
    plt.show()
    
    
    #plot imshow motion blur
    plt.imshow(img_motion_blur , cmap='gray')
    plt.show()
    
    
    #deconvolve the image 
    original_image = fft_deconvolution(img_motion_blur , kernel_matrix_motion_blur) 
    plt.imshow(original_image , cmap='gray')
    plt.show()