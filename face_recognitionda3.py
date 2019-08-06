# before starting this code pls make sure you have install the Face_recognition library  by the following comand 
# pip install face_recognition
import face_recognition
import cv2
import numpy as np

# import the above library where face_recognization is an predefined library which uses deep learning algortiham 

# The below command is the video capture used for videocamera opening 
video_capture = cv2.VideoCapture(0)

# below command is used for the sending the image to the deeplearning algoritham ----------->face_recogination
Har1_image = face_recognition.load_image_file("/Users/sheggamharshith/Desktop/python face recognization/new.png")

#here we have decoded the image into simple array that will be stored into the face_decoding variable
har1_face_encoding = face_recognition.face_encodings(Har1_image)[0]
print("the decoding of the image and saving it into the array ",har1_face_encoding)
#to find out what kind of the array it is we are giving the command type() we get numpy.narray as output
print('the type of array is ',type(har1_face_encoding))

# this is traning set 1(obama) 
har2_image = face_recognition.load_image_file("/Users/sheggamharshith/Desktop/python face recognization/face_video_recognization/sameera afreen.jpeg")
har2_face_encoding = face_recognition.face_encodings(har2_image)[0]

#this is traning set 2(sreven sir)
Har3_image = face_recognition.load_image_file("/Users/sheggamharshith/Desktop/python face recognization/Unknown-1 3.10.45 PM.jpeg")
har3_face_encoding = face_recognition.face_encodings(Har3_image)[0]

#this is traning set 3(rohith raja)
Har4_image = face_recognition.load_image_file("/Users/sheggamharshith/Desktop/python face recognization/Unknown-2 3.10.45 PM.jpeg")
har4_face_encoding = face_recognition.face_encodings(Har4_image)[0]

#this is traning set 4(da3)
Har5_image = face_recognition.load_image_file("/Users/sheggamharshith/Desktop/python face recognization/face_video_recognization/da3.jpeg")
har5_face_encoding = face_recognition.face_encodings(Har5_image)[0]

#this is traning set 5(da3)
Har6_image = face_recognition.load_image_file("/Users/sheggamharshith/Desktop/python face recognization/face_video_recognization/sameera afreen.jpeg")
har6_face_encoding = face_recognition.face_encodings(Har6_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
   har1_face_encoding,
   har2_face_encoding,
   har3_face_encoding,
   har4_face_encoding,
   har5_face_encoding,
   har6_face_encoding
]
known_face_names = [
    "harshith",
    "obama",
    "sraven sir",
    "rohith raja sir",
    "dha3",
    "Afeera samreen"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # here we are displaying the message in the seprate window using the app  
        #window = tk.Tk()
       # tk.Label(window, text = len(face_locations),fg = 'red',font = "Arial").pack(side = 'bottom')
        print(len(face_locations))
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
