import tensorflow as tf
from PIL import Image, ImageOps     # Install pillow instead of PIL
import numpy as np                  # Numpy provides numeric processing
import shutil
import datetime                     # What time is it for solar altitude
import time
import requests
import os

class MeteorDetection():
    def __init__(self):        
        np.set_printoptions(suppress=True) 

        # Create necessary directories
        os.makedirs("detections", exist_ok=True)
        os.makedirs("night_images", exist_ok=True)

        # Load the model 
        self.model = tf.keras.models.load_model(r"data\keras_model.h5", compile=False) 

        # Load the labels 
        self.class_names = open(r"data\labels.txt", "r").readlines()

        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) 

    def detect(self, path):
        try:
            # If the sun is up don't bother running the model 
            # (You can add solar altitude calculation here later)
            
            # Load the image from the AllSkyCam 
            image = Image.open(path).convert("RGB") 

            # resizing the image to be at least 224x224 cropped from the center 
            size = (224, 224) 
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS) 

            # turn the image into a numpy array 
            image_array = np.asarray(image) 

            # Normalize the image 
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1 

            # Load the image into the array 
            self.data[0] = normalized_image_array 

            # Run the image through the model 
            prediction = self.model.predict(self.data) 
            index = np.argmax(prediction) 
            class_name = self.class_names[index].strip() 
            confidence_score = prediction[0][index] 

            # Create unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            new_file_path = f"detections/meteor_{timestamp}.jpg"

            if index == 0 and confidence_score > 0.75:  # Assuming class 0 is "meteor"
                print(f"Meteor Detected: {class_name} ({confidence_score*100:.2f}%)")
                shutil.copy2(path, new_file_path)
                return True, confidence_score
            else:
                print(f"No meteor: {class_name} ({confidence_score*100:.2f}%)")
                return False, confidence_score
                
        except Exception as e:
            print(f"Error in detection: {e}")
            return False, 0.0

def download_image_url(url, identifier):   
    headers = {'User-Agent': 'Mozilla/5.0'} 
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=15, verify=False)
        if response.status_code == 200:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"night_images/allsky_{identifier}_{timestamp}.jpg"
            
            with open(save_path, 'wb') as out_file:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, out_file)
            
            print(f"Image downloaded: {save_path}")
            return save_path
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def is_night_time():
    """Check if it's currently night time to avoid processing during daylight"""
    current_hour = datetime.datetime.now().hour
    # Adjust these hours based on your location and season
    return current_hour >= 18 or current_hour <= 6

if __name__ == "__main__":
    meteor = MeteorDetection()
    print("Meteor Detection Started")
    
    # URLs to monitor
    allsky_urls = [
        "https://coopd.lna.br:8090/img/allsky_picole.jpg",
        "https://coopd.lna.br:8090/img/allsky340c.jpg"
    ]
    
    last_download_time = 0
    download_interval = 60  # Download every 60 seconds
    
    try:
        while True:
            current_time = time.time()
            
            # Check if it's time to download new images
            if current_time - last_download_time >= download_interval:
                if is_night_time():  # Only run at night
                    for i, url in enumerate(allsky_urls):
                        print(f"Downloading from AllSky {i+1}...")
                        img_path = download_image_url(url, f"cam{i+1}")
                        if img_path:
                            meteor.detect(img_path)
                    
                    last_download_time = current_time
                else:
                    print("Daytime - skipping detection")
                    time.sleep(300)  # Sleep longer during daytime
                    last_download_time = time.time()  # Reset timer
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Meteor Detection Stopped")
    except Exception as e:
        print(f"Unexpected error: {e}")