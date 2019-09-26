import os
import tempfile
from pdf2image import convert_from_path
import cv2
 
files = filter(os.path.isfile, os.listdir( os.curdir ) )
for singlefile in files:
    if_pdf=singlefile.split('.')
    if if_pdf[1]=='pdf':
        try:
            images_from_path = convert_from_path(singlefile, last_page=1, first_page =0)
            base_filename  =  if_pdf[0] + '.jpg'
            save_dir = os.getcwd()
            images_from_path[0].save(os.path.join(save_dir, base_filename), 'JPEG')
            img=cv2.imread(base_filename)
            dim = (299, 299)
            resizeImg=cv2.resize(img, dim)
            cv2.imwrite(base_filename,resizeImg)
            #os.remove(singlefile)
        except:
            print(singlefile)
    
print("Done!")

