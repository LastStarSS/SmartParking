import cv2

# Load your image
image = cv2.imread("layout.jpg")
clone = image.copy()
cropping = False
start_point = ()
end_point = ()

def mouse_crop(event, x, y, flags, param):
    global start_point, end_point, cropping, image

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp = clone.copy()
        cv2.rectangle(temp, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Crop Tool", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        cropping = False

        x1, y1 = start_point
        x2, y2 = end_point
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        cropped = clone[y_min:y_max, x_min:x_max]
        cv2.imshow("Cropped", cropped)
        cv2.imwrite("cropped.jpg", cropped)
        print(f"Top-left: ({x_min}, {y_min})")
        print(f"Bottom-right: ({x_max}, {y_max})")

cv2.namedWindow("Crop Tool")
cv2.setMouseCallback("Crop Tool", mouse_crop)

while True:
    cv2.imshow("Crop Tool", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):  # Reset the image
        image = clone.copy()
    elif key == ord("q"):  # Quit
        break

cv2.destroyAllWindows()
