import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

# Load the trained model to classify the sign
from keras.models import load_model
model = load_model('traffic_classifier.h5')

# Dictionary to label all traffic signs class.
classes = {
            (0,):'Speed limit (20km/h)',
            (1,):'Speed limit (30km/h)', 
            (2,):'Speed limit (50km/h)', 
            (3,):'Speed limit (60km/h)', 
            (4,):'Speed limit (70km/h)', 
            (5,):'Speed limit (80km/h)', 
            (6,):'End of speed limit (80km/h)', 
            (7,):'Speed limit (100km/h)', 
            (8,):'Speed limit (120km/h)', 
            (9,):'No overtaking', 
            (10,):'No passing by vehicles over 3.5 tons', 
            (11,):'Right-of-way at intersection', 
            (12,):'Priority road', 
            (13,):'Yield', 
            (14,):'Stop', 
            (15,):'No vehicles', 
            (16,):'Veh > 3.5 tons prohibited', 
            (17,):'No entry', 
            (18,):'General caution(Danger warning)', 
            (19,):'Dangerous curve left', 
            (20,):'Dangerous curve right', 
            (21,):'Double curve,First left then right', 
            (22,):'Bumpy road', 
            (23,):'Slippery road', 
            (24,):'Road narrows on the right', 
            (25,):'Road work', 
            (26,):'Traffic signals', 
            (27,):'Pedestrians crossing', 
            (28,):'Children crossing', 
            (29,):'Bicycles crossing', 
            (30,):'Beware of ice/snow',
            (31,):'Wild animals crossing', 
            (32,):'End speed + passing limits', 
            (33,):'Turn right ahead', 
            (34,):'Turn left ahead', 
            (35,):'Ahead only', 
            (36,):'Go straight or right', 
            (37,):'Go straight or left', 
            (38,):'Keep right', 
            (39,):'Keep left', 
            (40,):'Roundabout mandatory', 
            (41,):'End of no passing', 
            (42,):'End no passing vehicle > 3.5 tons'}

# Dictionary containing explanations for each traffic sign class.
explanations = {
            (0,):' Indicates a maximum speed of 20 kilometers per hour allowed in the designated area to ensure safety and reduce potential accidents.',
            (1,):' Indicates a maximum speed of 30 kilometers per hour allowed in the designated area, ensuring safer driving conditions and reducing the severity of potential collisions.', 
            (2,):' Indicates a maximum speed of 50 kilometers per hour allowed in the designated area to promote safe driving and reduce the risk of accidents.', 
            (3,):' Indicates a maximum speed of 60 kilometers per hour allowed in the designated area to maintain road safety and minimize the likelihood of accidents.', 
            (4,):'Indicates a maximum speed of 70 kilometers per hour allowed in the designated area to ensure safe and efficient traffic flow while reducing the risk of potential collisions.', 
            (5,):' Indicates a maximum speed of 80 kilometers per hour allowed on the road, promoting a balance between traffic efficiency and safety for drivers and other road users.', 
            (6,):'Indicates the end of the previously designated speed limit of 80 kilometers per hour, allowing drivers to resume normal speed limits in the area beyond this sign. Caution should be exercised to adapt to the changing road conditions.', 
            (7,):'Indicates a maximum speed of 100 kilometers per hour allowed on the road, promoting faster traffic flow on highways and well-designed roadways while emphasizing the need for increased attention and adherence to safety regulations.', 
            (8,):'indicates a maximum speed of 120 kilometers per hour allowed on the road, typically applicable to well-built highways, facilitating smoother traffic flow and demanding heightened vigilance from drivers to ensure safe and efficient travel.', 
            (9,):' indicates that overtaking or passing other vehicles is prohibited in the specified section of the road, ensuring safer driving conditions, preventing potential collisions, and maintaining a steady traffic flow.', 
            (10,):'"No Passing by Vehicles Over 3.5 Tons": This sign indicates that vehicles weighing more than 3.5 tons are not allowed to overtake or pass other vehicles in the designated area, promoting safer driving conditions and preventing potential hazards on the road.', 
            (11,):'indicates the priority of passage at an intersection, and the vehicles approaching the intersection from the direction of the sign have the right-of-way over vehicles approaching from other directions, promoting smoother traffic flow and reducing the risk of collisions.', 
            (12,):'This sign designates a road with priority over intersecting roads, and vehicles on this road have the right-of-way over vehicles approaching from other roads, ensuring smoother traffic flow and reducing the potential for accidents at intersections.', 
            (13,):'This sign indicates that drivers approaching the intersection or merging point must give the right-of-way to other vehicles already on the main road or in the traffic stream, promoting safe merging and avoiding traffic disruptions.', 
            (14,):'This sign requires drivers to come to a complete halt at the designated point, giving way to other vehicles and pedestrians before proceeding, ensuring safety at intersections and preventing accidents.', 
            (15,):' indicates that no motor vehicles are allowed to enter the designated area, ensuring the safety of pedestrians or preserving the integrity of a restricted zone.', 
            (16,):' restricts entry to certain vehicles, specifically those weighing more than 3.5 tons, such as heavy trucks and large commercial vehicles, from entering a particular road or area', 
            (17,):'indicates that the designated road or area is off-limits for all vehicles and pedestrians, and entry is strictly prohibited, typically due to safety or security reasons.', 
            (18,):'alert drivers of potential hazards ahead, such as sharp curves, slippery roads or other dangerous conditions for safety.', 
            (19,):' This warning sign informs drivers of an upcoming sharp curve to the left, indicating the need to reduce speed and navigate the curve with caution to avoid potential accidents or loss of control.', 
            (20,):'This warning sign informs drivers of an upcoming sharp curve to the right, indicating the need to reduce speed and navigate the curve with caution to avoid potential accidents or loss of control.', 
            (21,):'This warning sign alerts drivers to an upcoming section of road with two consecutive curves, first to the left and then to the right, indicating the need for increased caution and reduced speed to navigate the curves safely and avoid potential accidents.', 
            (22,):'This warning indicates an uneven road surface with irregularities, advising drivers to proceed with caution to ensure a smoother and safer journey.', 
            (23,):'This warning indicates that the road surface is slippery, advising drivers to exercise caution and reduce speed to prevent skidding or loss of control due to reduced traction', 
            (24,):'This warning indicates that the width of the road will decrease on the right side, prompting drivers to be cautious and potentially yield to oncoming traffic while passing through the narrowed section', 
            (25,):'This warning indicates ongoing road construction or maintenance ahead, advising drivers to exercise caution, reduce speed, and follow any instructions or lane changes to ensure safety and smooth traffic flow', 
            (26,):'Traffic signals" refer to the various traffic lights at intersections that control the flow of vehicles and pedestrians, providing instructions such as stop, go, and yield to ensure safe and orderly traffic movement.', 
            (27,):'This indicates a designated area where pedestrians have the right-of-way to cross the road, prompting drivers to stop and yield to pedestrians to ensure their safety while crossing the street."', 
            (28,):'this warning indicates a designated area where children may be crossing the road, prompting drivers to exercise extreme caution, reduce speed, and be prepared to stop to ensure the safety of young pedestrians', 
            (29,):'indicates a designated area where bicycles may be crossing the road, prompting drivers to exercise caution, reduce speed, and be prepared to yield to cyclists to ensure their safety while crossing', 
            (30,):'This warning advises drivers to be cautious and watch out for icy or snowy road conditions, prompting them to reduce speed and drive carefully to prevent skidding or loss of control due to slippery surfaces',
            (31,):'indicates a potential presence of wild animals crossing the road, prompting drivers to be vigilant, reduce speed, and exercise caution to avoid collisions and ensure the safety of both the animals and the passengers.', 
            (32,):'It indicates the end of the speed limit and passing restrictions, allowing drivers to resume regular speed and pass other vehicles if permitted on the road.', 
            (33,):' indicates that there is a right turn ahead on the road, prompting drivers to be prepared to make a right turn and to slow down if necessary', 
            (34,):'indicates that there is a left turn ahead on the road, prompting drivers to be prepared to make a left turn and to slow down if necessary', 
            (35,):'indicates that the road ahead leads only in the direction shown, with no alternative routes, prompting drivers to proceed straight without making any turns.', 
            (36,):'It indicates that the road ahead allows drivers to go straight or make a right turn, prompting them to choose either direction depending on their intended route.', 
            (37,):' indicates that the road ahead allows drivers to go straight or make a left turn, prompting them to choose either direction depending on their intended route.', 
            (38,):'indicates that drivers should keep to the right side of the road or stay in the right lane, following the traffic flow and potentially making a right turn or staying on the right side of the road', 
            (39,):'indicates that drivers should keep to the left side of the road or stay in the left lane, following the traffic flow and potentially making a left turn or staying on the left side of the road', 
            (40,):'indicates that entering the roundabout is mandatory for all drivers approaching it, requiring them to yield to vehicles already inside the roundabout and proceed in a counterclockwise direction', 
            (41,):' indicates the end of the section where passing or overtaking other vehicles was prohibited, allowing drivers to resume passing if it is safe and legally permitted.', 
            (42,):' indicates the end of the restriction that prohibited vehicles over 3.5 tons from passing, allowing them to pass other vehicles if it is safe and legally permitted.'}

# Function to classify an image
def classify_image(file_path, label, explanation_label):
    try:
        image = Image.open(file_path)
        image = image.resize((30, 30))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)

        pred = model.predict(image)
        pred_class = np.argmax(pred, axis=1)
        pred_tuple = tuple(pred_class)

        if pred_tuple in classes:
            sign = classes[pred_tuple]
            label.configure(foreground='red', text=sign)  # Set class color to red
            explanation = explanations[pred_tuple]
            explanation_label.configure(foreground='blue', text=explanation, wraplength=900)  
        else:
            sign = "Unknown sign"
            label.configure(foreground='red', text=sign)
            explanation_label.configure(foreground='gray', text='')

    except Exception as e:
        print(f"Error: {e}")

# Function to handle the "Classify Images" button click
def classify_button_click():
    classify_image(file_path_1, label_1, explanation_label_1)
    classify_image(file_path_2, label_2, explanation_label_2)
    classify_image(file_path_3, label_3, explanation_label_3)

# Function to handle image upload for image 1
def upload_image_1():
    global file_path_1
    file_path_1 = filedialog.askopenfilename()
    uploaded = Image.open(file_path_1)
    uploaded.thumbnail((180, 180))  # Adjust the thumbnail size (3cm) as needed
    im = ImageTk.PhotoImage(uploaded)

    sign_image_1.configure(image=im)
    sign_image_1.image = im
    label_1.configure(text='')
    explanation_label_1.configure(text='')

# Function to handle image upload for image 2
def upload_image_2():
    global file_path_2
    file_path_2 = filedialog.askopenfilename()
    uploaded = Image.open(file_path_2)
    uploaded.thumbnail((180, 180))  # Adjust the thumbnail size (3cm) as needed
    im = ImageTk.PhotoImage(uploaded)

    sign_image_2.configure(image=im)
    sign_image_2.image = im
    label_2.configure(text='')
    explanation_label_2.configure(text='')

# Function to handle image upload for image 3
def upload_image_3():
    global file_path_3
    file_path_3 = filedialog.askopenfilename()
    uploaded = Image.open(file_path_3)
    uploaded.thumbnail((180, 180))  # Adjust the thumbnail size (3cm) as needed
    im = ImageTk.PhotoImage(uploaded)

    sign_image_3.configure(image=im)
    sign_image_3.image = im
    label_3.configure(text='')
    explanation_label_3.configure(text='')

# Function to center the window on the screen
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")

# Initialise GUI
top = tk.Tk()
top.title('Traffic Sign Recognition')


# Center the GUI window on the screen
center_window(top, 800, 600)

# Background Image (Modify the path according to your image location)
background_image = Image.open(r"C:\Users\MANJU\Desktop\Traffic Project\bg.jpg")
bg_width, bg_height = top.winfo_screenwidth(), top.winfo_screenheight()
background_image = background_image.resize((bg_width, bg_height), Image.ANTIALIAS)
background_photo = ImageTk.PhotoImage(background_image)
background_label = Label(top, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Creating sign_image and label widgets for image 1
sign_image_1 = Label(top)
label_1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
explanation_label_1 = Label(top, background='#CDCDCD', font=('arial', 12))

# Creating sign_image and label widgets for image 2
sign_image_2 = Label(top)
label_2 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
explanation_label_2 = Label(top, background='#CDCDCD', font=('arial', 12))

# Creating sign_image and label widgets for image 3
sign_image_3 = Label(top)
label_3 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
explanation_label_3 = Label(top, background='#CDCDCD', font=('arial', 12))


# Buttons for image uploads
upload_1 = Button(top, text="Upload Image 1", command=upload_image_1, padx=10, pady=5)
upload_1.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload_2 = Button(top, text="Upload Image 2", command=upload_image_2, padx=10, pady=5)
upload_2.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload_3 = Button(top, text="Upload Image 3", command=upload_image_3, padx=10, pady=5)
upload_3.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

# Button to classify images
classify_b = Button(top, text="Classify Images", command=classify_button_click, padx=10, pady=5)
classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

# Create a green heading label
heading = Label(top, text="Traffic Sign Recognition", font=('arial', 15, 'bold'), foreground='dark blue', background='#0cf537')
heading.pack()

# Layout using grid
upload_1.pack(side=LEFT)  # Add 3cm distance between heading and buttons
upload_2.pack(side=LEFT)
upload_3.pack(side=LEFT)
classify_b.pack(pady=(20, 0))  # Add 3cm distance between buttons and images
sign_image_1.pack(pady=(0, 20))
label_1.pack()
explanation_label_1.pack()
sign_image_2.pack(pady=(0, 20))
label_2.pack()
explanation_label_2.pack()
sign_image_3.pack(pady=(0, 20))
label_3.pack()
explanation_label_3.pack()

Manju_info= Label(top, text="\n Project by:\nName     : S.Manjulatha\nId      : N180490\nContact : 8897586533\n Email : n180490@rguktn.ac.in\n\n\n ", font=('arial', 10, 'bold'), foreground='blue', background='#05f1f5')
Manju_info.place(x=10, y=top.winfo_screenheight() - 100, anchor="w")

top.mainloop()