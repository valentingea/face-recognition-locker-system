import os
import streamlit as st
import cv2
import time
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import base64

# Data loker
lockers_data = [
    {"name": "A1", "value": "available"},
    {"name": "A2", "value": "available"},
    {"name": "A3", "value": "available"},
    {"name": "A4", "value": "available"},
    {"name": "A5", "value": "available"},
    {"name": "A6", "value": "available"},
    {"name": "A7", "value": "available"},
    {"name": "A8", "value": "available"}
]

all = ["A1","A2","A3","A4","A5","A6","A7","A8"]
#used = ['A2', 'A3']

folder_path = 'data'
all_folders = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

# Mendapatkan daftar folder yang berisi gambar
used = []
for folder in all_folders:
    folder_content = os.listdir(os.path.join(folder_path, folder))
    if any(image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) for image_file in folder_content):
        used.append(folder)

available = [locker for locker in all if locker not in used]

menu_choice = st.sidebar.radio(
        "Main Menu", 
        ['🏠 Home', '🗄️ New Locker', '🔓 Reopen', '✅ Finish Use', 'Test Image'],
        captions=["Homepage", "First time use the locker", "Reopen locker that you used", "Finish used your locker", "Just for testing"])

def home():
    st.subheader("Welcome to **FaceFortify** : Face Recognition Security System")

    st.write(
        "FaceFortify is a Face Recognition Security System that provides secure locker access using facial recognition technology."
    )

    st.write("### Getting Started:")
    st.write(
        "1. **New Locker**: Open a new locker by selecting an available locker, entering your name, and capturing your face picture."
    )
    st.write(
        "2. **Reopen Locker**: Reopen your locker by selecting the locker you previously used"
    )
    st.write(
        "3. **Finish Use**: Finish using your locker by selecting the locker you want to finish using"
    )

    st.write("### Tips for Better Recognition:")
    st.write(
        "1. Ensure good lighting conditions for accurate face detection."
    )
    st.write(
        "2. Position your face properly in front of the camera during image capture."
    )
    st.write(
        "3. Use a clear and recent face picture for better recognition accuracy."
    )

    st.write("### Important Note:")
    st.write(
        "This system relies on facial recognition technology. Please make sure your face is properly captured for secure locker access."
    )

def check(image_path):
    device = torch.device("cpu")
    num_classes = 5

    # Path to the model that has been saved
    model_folder = 'train_model'
    model_path = os.path.join(model_folder, 'model.pth')

    # Load the pre-trained VGG16 model
    vgg_test = models.vgg16()
    vgg_test.classifier[-1] = nn.Linear(vgg_test.classifier[-1].in_features, num_classes)

    # Load the saved model parameters
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    vgg_test.load_state_dict(checkpoint['model_state_dict'])

    # Access the class_names attribute from the loaded model
    class_names = checkpoint['class_names']

    # Set class_names as an attribute of vgg_test
    vgg_test.class_names = class_names

    # Move the model to the appropriate device
    vgg_test = vgg_test.to(device)

    # Ensure the model is in evaluation mode
    vgg_test.eval()

    # Define the transformation for test images
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Function to perform testing on a single image
    def test_single_image(image_path, model):
        # Read the image
        image = Image.open(image_path).convert("RGB")

        # Apply the transformation
        image = test_transform(image)

        # Add a batch dimension
        image = image.unsqueeze(0).to(device)

        # Get the output from the model
        with torch.no_grad():
            output = model(image)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)

        # Get the name of the predicted class from the model's attribute
        class_name = model.class_names[predicted_class.item()]

        return class_name

    predicted_class = test_single_image(image_path, vgg_test)

    return predicted_class

# Fungsi untuk membuka loker baru
def open_new_locker():
    global all
    st.subheader("Open New Locker")

    for i in range(2):
        # Membagi setiap baris menjadi 4 kolom
        locs = st.columns(4)

        for j, (loc, locker_data) in enumerate(zip(locs, lockers_data[i*4:(i+1)*4])):
            # Mengekstrak informasi dari data
            locker_name = locker_data["name"]
            locker_value = locker_data["value"]

            if locker_name in st.session_state.used:
                locker_value = "-in use"

            # Membuat metric
            loc.metric("Locker", locker_name, locker_value)

    selected_locker = st.selectbox("Select an available locker", ("",) + tuple(st.session_state.available))
    st.write('You selected:', selected_locker)

    name = st.text_input("Enter name")
    
    # Jalankan kamera untuk mengambil gambar wajah
    st.write("##### Face the camera and press the 'Take Picture' button")
    camst = st.camera_input('Take Picture')

    if camst is not None:
        path = os.path.dirname(os.path.abspath(__file__))
        cam = cv2.VideoCapture(0)
        detector=cv2.CascadeClassifier(path+r'\face_detect\face.xml')
        i=0
        offset=50
        collect = True

        my_bar = st.progress(0, "Collecting photos. Please wait for a moment...")

        while collect == True:
            ret, im =cam.read()
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in faces:
                i=i+1

                # Potongan gambar warna
                face_color = im[y-offset:y+h+offset, x-offset:x+w+offset]

                cv2.imwrite("data/"+ selected_locker + '/' + name + '_' + str(i) + ".jpg", face_color)
                cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
                cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
                cv2.waitKey(100)

                time.sleep(0.01)
                my_bar.progress(i*3, "Collecting photos. Please wait for a moment...")

            if i>=30:
                cam.release()
                cv2.destroyAllWindows()

                my_bar.progress(i*3, "Finish Collecting photos")

                text = "You can use your locker "+selected_locker+" now"
                st.success(text)

                st.session_state.used.append(selected_locker)
                st.session_state.available = [locker for locker in all if locker not in st.session_state.used]
                    
                collect = False
                break

# Fungsi untuk membuka kembali loker
def reopen_locker(model):
    st.subheader("Reopen Locker")

    for i in range(3):
        # Membagi setiap baris menjadi 4 kolom
        locs = st.columns(4)

        for j, (loc, locker_data) in enumerate(zip(locs, lockers_data[i*4:(i+1)*4])):
            # Mengekstrak informasi dari data
            locker_name = locker_data["name"]
            locker_value = locker_data["value"]

            if locker_name in st.session_state.used:
                locker_value = "-in use"

            # Membuat metric
            loc.metric("Locker", locker_name, locker_value)

    selected_locker = st.selectbox("Select the locker you use", ("",) + tuple(st.session_state.used))
    st.write('You selected:', selected_locker)

    name = st.text_input("Enter name")

    label = f"{selected_locker}_{name}"
    
    # Jalankan kamera untuk mengambil gambar wajah
    st.write("##### Face the camera and press the 'Take Picture' button")
    camst = st.camera_input("Take a picture")

    if camst is not None:       
        load = st.spinner('Searcing data...')
        with load:
            predict = check(camst)
        if predict == label:
            st.success("Your locker has successfully open")
        else:
            st.warning('Your data is not matched', icon="⚠️")

# Fungsi untuk menyelesaikan penggunaan loker
def finish(model):
    global all
    st.subheader("Finish use")

    for i in range(3):
        # Membagi setiap baris menjadi 4 kolom
        locs = st.columns(4)

        for j, (loc, locker_data) in enumerate(zip(locs, lockers_data[i*4:(i+1)*4])):
            # Mengekstrak informasi dari data
            locker_name = locker_data["name"]
            locker_value = locker_data["value"]

            if locker_name in st.session_state.used:
                locker_value = "-in use"

            # Membuat metric
            loc.metric("Locker", locker_name, locker_value)

    selected_locker = st.selectbox("Select the locker you use", ("",) + tuple(st.session_state.used))
    st.write('You selected:', selected_locker)

    name = st.text_input("Enter name")

    label = f"{selected_locker}_{name}"
    
    # Jalankan kamera untuk mengambil gambar wajah
    st.write("##### Face the camera and press the 'Take Picture' button")
    camst = st.camera_input("Take a picture")

    if camst is not None:       
        load = st.spinner('Searcing data...')
        with load:
            predict = check(camst)
        if predict == label:
            st.success("Your locker has successfully open")
            st.session_state.used.remove(selected_locker)
            st.session_state.available = [locker for locker in all if locker not in st.session_state.used]
        else:
            st.warning('Your data is not matched', icon="⚠️")            

def page_test_image():
    device = torch.device("cpu")
    num_classes = 5

    # Path to the model that has been saved
    model_folder = 'train_model'
    model_path = os.path.join(model_folder, 'model.pth')

    # Load the pre-trained VGG16 model
    vgg_test = models.vgg16()
    vgg_test.classifier[-1] = nn.Linear(vgg_test.classifier[-1].in_features, num_classes)

    # Load the saved model parameters
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    vgg_test.load_state_dict(checkpoint['model_state_dict'])

    # Access the class_names attribute from the loaded model
    class_names = checkpoint['class_names']

    # Set class_names as an attribute of vgg_test
    vgg_test.class_names = class_names

    # Move the model to the appropriate device
    vgg_test = vgg_test.to(device)

    # Ensure the model is in evaluation mode
    vgg_test.eval()

    # Define the transformation for test images
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Function to perform testing on a single image
    def test_single_image(image_path, model):
        # Read the image
        image = Image.open(image_path).convert("RGB")

        # Apply the transformation
        image = test_transform(image)

        # Add a batch dimension
        image = image.unsqueeze(0).to(device)

        # Get the output from the model
        with torch.no_grad():
            output = model(image)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)

        # Get the name of the predicted class from the model's attribute
        class_name = model.class_names[predicted_class.item()]

        return class_name

    # Path to the folder containing test images
    test_images_folder = 'new_test'

    # Test multiple images
    for image_file in os.listdir(test_images_folder):
        image_path = os.path.join(test_images_folder, image_file)
        predicted_class = test_single_image(image_path, vgg_test)

        # Display image using HTML
        st.markdown(f'<img src="data:image/jpeg;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" alt="image">', unsafe_allow_html=True)
        st.write(f"Predicted : {predicted_class}")

def main():
    st.title("FaceFortify")

    if 'used' not in st.session_state:
        st.session_state.used = []
        for folder in all_folders:
            folder_content = os.listdir(os.path.join(folder_path, folder))
            if any(image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) for image_file in folder_content):
                st.session_state.used.append(folder)

    if 'available' not in st.session_state:
        st.session_state.available = [locker for locker in all if locker not in st.session_state.used]
    
    # Model wajah (ganti dengan model wajah yang sesuai)
    face_model = None  # Ganti dengan model wajah yang akan Anda gunakan

    if menu_choice == '🏠 Home':
        home()
    elif menu_choice == '🗄️ New Locker':
        open_new_locker()
    elif menu_choice == '🔓 Reopen':
        reopen_locker()
    elif menu_choice == '✅ Finish Use':
        finish()
    else:
        page_test_image()

if __name__ == "__main__":
    main()
