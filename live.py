# This script will detect faces via your webcam.
# Tested with OpenCV3

import face_recognition
import cv2
import numpy

cap = cv2.VideoCapture(0)

# Create the haar cascade
# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load a sample picture and learn how to recognize it.
sample_image = face_recognition.load_image_file("faces/me.jpg")
sample_face_encoding = face_recognition.face_encodings(sample_image)[0]

# sample_face_encoding = numpy.loadtxt('test3.txt')

known_face_encodings = [
	sample_face_encoding
]

known_face_names = [
    "Khong Phalla"
	# "Khim Kheng"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
capture_fram = False

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# flip video image vertically
	# frame = cv2.flip(frame, -1)

	if capture_fram:
		# Save the captured image into the datasets folder
		cv2.imwrite("faces/user2.jpg", frame)
		capture_fram = False

	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]
	# rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

	# Only process every other frame of video to save time
	if process_this_frame:
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
		
		face_names = []
		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			
			name = "Unknown"
			
			# If a match was found in known_face_encodings, just use the first one.
			# if True in matches:
			# 	first_match_index = matches.index(True)
			# 	name = known_face_names[first_match_index]
			
			# Or instead, use the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = numpy.argmin(face_distances)
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

	# # Our operations on the frame come here
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# # Detect faces in the image
	# faces = faceCascade.detectMultiScale(
	# 	gray,
	# 	scaleFactor=1.1,
	# 	minNeighbors=5,
	# 	minSize=(30, 30)
	# 	#flags = cv2.CV_HAAR_SCALE_IMAGE
	# )

	# print("Found {0} faces!".format(len(faces)))

	# # Draw a rectangle around the faces
	# for (x, y, w, h) in faces:
	# 	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


	# Display the resulting frame
	cv2.imshow('frame', frame)

	# Get key press q for exit and c for capture image
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
	elif key == ord('c'):
		# fl = open("demo_file.txt", "a")
		# fl.write(face_encodings)
		# fl.close()

		numpy.savetxt('test2.txt', face_encodings[0], delimiter=',')
		
		# # Save the captured image into the datasets folder
		# cv2.imwrite("faces/user2.jpg", frame)
		capture_fram = True

	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
