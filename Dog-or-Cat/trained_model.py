from keras.models import model_from_json
import cv2
import numpy as np

json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("./model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

image = cv2.imread('./test1/1.jpg')
image = cv2.resize(image, (50,50))
img = image.reshape(1, 50, 50, 3)

cv2.imshow("Input Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = loaded_model.predict_classes(img)
if(result[0][0] == 1):
    print("the image is of a Dog, probably")
else:
    print("the image is of a Cat, probably")