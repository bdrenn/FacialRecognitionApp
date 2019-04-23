import face_recognition

image_of_bill = face_recognition.load_image_file('./img/known/Bill_Gates.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

#Image of Bill Gates... Should return true
unknown_image = face_recognition.load_image_file('./img/unknown/bill-gates-4.jpg')
#Not image of Bill Gates... Should return false
unknown_image2 = face_recognition.load_image_file('./img/unknown/d-trump.jpg')

unknown_face_encoding = face_recognition.face_encodings(unknown_image2)[0]

#compare faces
results = face_recognition.compare_faces([bill_face_encoding], unknown_face_encoding)

if results[0]:
    print('This is Bill Gates')
else:
    print('This is not Bill Gates')

