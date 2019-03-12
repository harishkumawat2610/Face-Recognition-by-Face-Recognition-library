#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import cv2
import pandas as pd



# In[2]:


video_capture = cv2.VideoCapture(0)


# In[3]:


#me
hk_image = face_recognition.load_image_file("hk.jpg")
hk_face_encoding = face_recognition.face_encodings(hk_image)[0]


# In[4]:


#khorwal
kho_image = face_recognition.load_image_file("kho.jpg")
kho_face_encoding = face_recognition.face_encodings(kho_image)[0]


# In[5]:


#jyoti mem
jg_image = face_recognition.load_image_file("jg.jpg")
jg_face_encoding = face_recognition.face_encodings(jg_image)[0]


# In[6]:


#vinesh jain sir
vj_image = face_recognition.load_image_file("vj.jpg")
vj_face_encoding = face_recognition.face_encodings(vj_image)[0]


# In[7]:


#vishnu sir
vps_image = face_recognition.load_image_file("vps.jpg")
vps_face_encoding = face_recognition.face_encodings(vps_image)[0]


# In[8]:


#tazi  sir
tz_image = face_recognition.load_image_file("tz.jpg")
tz_face_encoding = face_recognition.face_encodings(tz_image)[0]


# In[9]:


known_face_encodings = [
    hk_face_encoding,
    kho_face_encoding,
    jg_face_encoding,
    vj_face_encoding,
    vps_face_encoding,
    tz_face_encoding
    
]
known_face_names = [
    "Harish",
    "Hemant",
    "Jyoti Mem",
    "Vinesh Sir",
    "VPS Sir",
    "Sawra Uncle"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
row = 0
column = 0

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

        face_names = []
        for face_encoding in face_encodings:
            
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                
                

            face_names.append(name)
           
  
        # incrementing the value of row by one 
        # with each iteratons. 
            
      
        #workbook.close() 

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






