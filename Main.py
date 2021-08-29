import tkinter as tk 
from PIL import ImageTk, Image,ImageOps
from tkinter import filedialog
import os
import numpy as np
import cv2 

class HomomorphicFilter:
    def __init__(self, gH= 1.5, gL= 0.5):
        self.gH = float(gH)
        self.gL = float(gL)

    def __Duv(self, I_shape):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)**(1/2)).astype(np.dtype('d'))
        return Duv
    
    # Butterworth Filters
    def __butterworth_filter(self, I_shape, filter_params):
        Duv=self.__Duv(I_shape)
        n=filter_params[2]
        c=filter_params[1]
        D0=filter_params[0]
        h = 1/(1+((c*Duv)/D0)**(2*n))
        H=(1-h)
        return H

    #Gaussian filter
    def __gaussian_filter(self, I_shape, filter_params):
        Duv=self.__Duv(I_shape)
        c=filter_params[1]
        D0=filter_params[0]
        h = np.exp((-c*(Duv**2)/(2*(D0**2))))#lowpass filter
        H=(1-h)
        return H
    
    # Methods
    def __apply_filter(self, I, H,params ):
        if self.gH<1 or self.gL>=1:
            H = H
        else:
            H = ((self.gH-self.gL)*H+self.gL)
        I_filtered=H*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H = None):
        #  Validating image as grayscale
        if len(I.shape) != 2:
            raise Exception('Improper image')
        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype='d') )
        I_fft = np.fft.fft2(I_log)
        I_fft=np.fft.fftshift(I_fft)
        # Applying Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        else:
            raise Exception('Selected filter not implemented')
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H, params=filter_params)
        I_fft_filt=np.fft.fftshift(I_fft_filt)
        I_filt = np.fft.ifft2(I_fft_filt)
        I=np.expm1(np.real(I_filt))
        
        Imax=(np.max(I))
        Imin=(np.min(I))
        I=255*((I-Imin)/(Imax-Imin))#Image is normalized
        return I

def openfn():
    filename = tk.filedialog.askopenfilename(title='open')
    return filename
    
def applyFilter():
    global number1, number2, number3,v
    a =number1.get()
    b =number2.get()
    c = number3.get()
    gH = 1.5
    gL = 0.5
    typeOfMethod = v.get()

    filter_value = [float(a),float(b),float(c)]
    x = openfn()
    img = Image.open(x)
    img = ImageOps.grayscale(img)
    arr_image = np.asarray(img)

    homo_filter = HomomorphicFilter( float(gH), float(gL))
    img_filtered = homo_filter.filter(I=arr_image, filter_params=[float(a),float(b),float(c)],filter= typeOfMethod)
    cv2.imwrite('filtered.png',(img_filtered))

    img.thumbnail((900, 800))
    img = ImageTk.PhotoImage(img)
    
    panel = tk.Label(root, image=img)
    panel.image = img
    panel.grid(row=3, column=0,columnspan=6,pady=(5, 20))

    label1 = tk.Label(root, text="Original Image",font = ("arial", 20, "bold")).grid(row=2, column=0,columnspan=6,pady=(20, 5))  
    label2 = tk.Label(root, text="Filtered Image",font = ("arial", 20, "bold")).grid(row=2, column=7,columnspan=6,pady=(20, 5))  

    load = Image.open('filtered.png')
    load.thumbnail((900, 800))
    filter_img = ImageTk.PhotoImage(load)
    panel2 = tk.Label(root, image=filter_img)
    panel2.image = filter_img
    panel2.grid(row=3, column=7,columnspan=6,pady=(5, 20))

root = tk.Tk()   
root.title('Homomorphic filtering Software')   
root.minsize(150, 400)

number1 = tk.StringVar(value = "50")  
number2 = tk.StringVar(value = "0.1")  
number3 = tk.StringVar(value = "0.2") 
v = tk.StringVar(value="1") 

labelNum1 = tk.Label(root, text="Cutoff Freq:").grid(row=1, column=0,padx = (30,5))  
entryNum1 = tk.Entry(root, textvariable=number1).grid(row=1, column=1)  

labelNum2 = tk.Label(root, text="Sharpness of Filter:").grid(row=1, column=2,padx = (30,5))  
entryNum2 = tk.Entry(root, textvariable=number2).grid(row=1, column=3)  

labelNum3 = tk.Label(root, text="Order of Filter:").grid(row=1, column=4,padx = (30,5))  
entryNum3 = tk.Entry(root, textvariable=number3).grid(row=1, column=5) 

tk.Radiobutton(root, text = "Butterworth", variable = v, value = 'butterworth').grid(row=1, column=10,padx = (30,5))  
tk.Radiobutton(root, text = "Gaussian", variable = v, value = 'gaussian').grid(row=1, column=11)  

entryNum4 = tk.Button(root, text='Select Image', command = applyFilter).grid(row=1, column=12,padx = 5)

root.mainloop()  