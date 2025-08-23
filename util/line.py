import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
# import isect_segments_bentley_ottmann.poly_point_isect as bot

def line1():
    img = cv2.imread("layout/2.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    cv2.imshow("line1", lines_edges)
    cv2.waitKey(0)
def line2():
    # white color mask
    img = cv2.imread("layout/1.jpg")
    #converted = convert_hls(img)
    image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = img.copy()
    cv2.imshow("mask",mask) 

def line3():
    # Load the image
    image = cv2.imread("layout/1.jpg")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a mask for white/yellow parking lines (tweak as needed)
    lower = np.array([0, 0, 200])
    upper = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and draw straight contours (likely lines)
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if cv2.arcLength(cnt, True) > 30 and len(approx) == 2:  # rough straight line filter
            x1, y1 = approx[0][0]
            x2, y2 = approx[1][0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show results
    cv2.imshow("Detected Lines", image)
    # cv2.imwrite("better_detected_lines.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def line4():
    img = cv2.imread('layout/1.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    print(lines)
    points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    print(lines_edges.shape)
    #cv2.imwrite('line_parking.png', lines_edges)

    print (points)
    intersections = bot.isect_segments(points)
    print (intersections)

    for inter in intersections:
        a, b = inter
        for i in range(3):
            for j in range(3):
                lines_edges[int(b) + i, int(a) + j] = [0, 255, 0]

    cv2.imshow('line_parking.png', lines_edges)


line1()



