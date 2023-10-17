import cv2
import numpy as np
import extcolors
import matplotlib.pyplot as plt
from PIL import Image

adjacent_range=2


#give input path of the image and an output path to write to
path = "D:/color-contrast-analyzer-version_2_2/sample/input/2.jpg"
output_path = "D:/color-contrast-analyzer-version_2_2/sample/output/new_gradient/2.jpg"
neighbouringPixels = []
filtered_contour=[]


def checkNeighbour (path, output_path) :
    color_tolerance=20
    min_contour_area = 400

    img_url = path
    # img = plt.imread(img_url)
    img = cv2.imread(img_url, cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(cv2.imread(img_url), cv2.COLOR_BGR2RGB)
    # # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img

    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 70, 210)

    # Find contours in the binary edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colors_x = extcolors.extract_from_path(img_url, tolerance = color_tolerance, limit = 30)
    all_color=colors_x

    #Logic for color tolerance
    k=100*(colors_x[0][0][1]+colors_x[0][2][1]+colors_x[0][3][1])/colors_x[1]
    if k<=80 and k>60:
        color_tolerance=20
        colors_x = extcolors.extract_from_path(img_url, tolerance = color_tolerance, limit = 30)
        Gradient_check_threshold=50
    if k<=55:
        color_tolerance=25
        colors_x = extcolors.extract_from_path(img_url, tolerance =color_tolerance, limit = 30)
        Gradient_check_threshold=60

    # removing colors which having low pixel count and can be gradient color and below code is to speed up process 
    #update( version 2.1)############################################################################################################

    color_tolerance = 1 if colors_x[1] <= 100000 else 5 if 100000 < colors_x[1] <= 300000 else \
                        10 if 300000 < colors_x[1] <= 500000 else 15

    black_white_colors = []
    remaining_colors = []
    total_remaining_pixels = 0

        # Extract black and white colors within the threshold
    for color, count in colors_x[0]:
        r, g, b = color
        if (r <= 50 and g <= 50 and b <= 50) or (r >= 225 and g >= 225 and b >= 225):
            black_white_colors.append((color, count))
        else:
            remaining_colors.append((color, count))
            total_remaining_pixels += count

    colors_x = (remaining_colors, total_remaining_pixels)  # Remove extracted colors from colors_x

    threshold = 0.001 * colors_x[1]  # Calculate the threshold count based on 0.1 percent to remove gradient color
    filtered_colors = [color for color in colors_x[0] if color[1] >= threshold]  # Filter out colors with count less than the threshold
    colors_x = (filtered_colors, sum(count for _, count in filtered_colors))  # Update colors_x with the filtered colors and count
    colors_x_appended = colors_x[0] + black_white_colors
    total_count = colors_x[1] + sum(count for _, count in black_white_colors)  # Calculate the total count including black and white colors
    colors_x = (colors_x_appended, total_count)  # Update colors_x with the appended colors and the total count
    all_colors = black_white_colors + [colors_x]
    total_count = sum(count for _, count in all_colors)

    combined_colors = list(colors_x)
    combined_colors.append(total_count)

    combined_colors.pop(-1)


    colors_x = combined_colors

    sorted_colors_x = sorted(colors_x[0], key=lambda x: x[1], reverse=True)
    colors_x[0] = sorted_colors_x
    print(colors_x)

    # find contours and check their colors, filter out the contours and colors which have a minimum area below min_contour_areas

    image = cv2.imread(path)
    img = Image.open(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_thresholds=20
    diff_threshold = 50
    min_contour_areas = 200
    filteredColors = colors_x[0].copy()
    
    print(len(colors_x[0]))
    for input_colors in range (0,len(colors_x[0])):
    
        target_color = colors_x[0][input_colors][0]
        color_threshold = color_thresholds  # Adjust this value based on your color similarity criteria
        color_diffs = np.abs(image_rgb - target_color)
        color_distances = np.sqrt(np.sum(color_diffs ** 2, axis=2))
        mask = color_distances <= color_threshold
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = min_contour_areas # Adjust this value based on your requirements
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
        if len(filtered_contours)>0: filtered_contour.append(filtered_contours)
        else : filteredColors.remove(colors_x[0][input_colors])
        print(len(filtered_contour))
    


    print(len(filtered_contour))
    len(filteredColors[0])
    diff_threshold = 50

    #reshape the filtered_contours list and find the colors of the neighbouring pixels around the coordinates, and keep them as a tuple in the neighbouringPixels array
    for color, contour in zip(filteredColors,filtered_contour) :
        print('finding target color:    ',color[0])
        target_color = color[0]
        contour_points = contour[0].reshape(-1, 2)
    #     print(contour_points)
        neighPix = []
        for coordinates in contour_points:
            x = coordinates[0]
            y = coordinates[1]
            pixelPos = (x,y)
            for neighPixel in range(1,4):
                try:
                    left_neigh_pixel = (y,x-neighPixel)
                    right_neigh_pixel = (y,x+neighPixel)
                    top_neigh_pixel = (y+neighPixel,x)
                    bottom_neigh_pixel = (y-neighPixel,x)
            
                    if all(x > 0 and y > 0 for x, y in [left_neigh_pixel, right_neigh_pixel, top_neigh_pixel, bottom_neigh_pixel]) :
    #                     finding colors of above neighbouring pixels
                        left_neigh_color = img.getpixel(left_neigh_pixel)[0:3]
                        right_neigh_color = img.getpixel(right_neigh_pixel)[0:3]
                        top_neigh_color = img.getpixel(top_neigh_pixel)[0:3]
                        bottom_neigh_color = img.getpixel(bottom_neigh_pixel)[0:3]
    
                                
                        if np.median(np.abs(np.array(left_neigh_color)-np.array(target_color))) <=diff_threshold :
                            neighPix.append(left_neigh_pixel)
                        else :
                            break
                                
                        if np.median(np.abs(np.array(right_neigh_color)-np.array(target_color))) <=diff_threshold :
                            neighPix.append(right_neigh_pixel)
                        else :
                            break
                                
                        if np.median(np.abs(np.array(top_neigh_color)-np.array(target_color))) <=diff_threshold :
                            neighPix.append(top_neigh_pixel)
                        else :
                            break
                                
                        if np.median(np.abs(np.array(bottom_neigh_color)-np.array(target_color))) <=diff_threshold :
                            neighPix.append(bottom_neigh_pixel)
                        else :
                            break
                        
                except Exception as e:
    #                 print(e)
                    pass
            if len(neighPix)>0:neighbouringPixels.append((neighPix, target_color))
    

    #print the length of the negihbouringPixels array
    print(len(neighbouringPixels))
    
    for pixelTuple in neighbouringPixels :
        coordinates = pixelTuple[0]
        color = pixelTuple[1]
        color_rgb = (color[2], color[1], color[0])
        #x,y works for some, gives out of bound error for some
        for coordinate in coordinates:
            x,y = coordinate
            image[x,y] = color_rgb

    #write the final changes to the image
    cv2.imwrite(output_path, image)




def call_main():
    checkNeighbour(path=path, output_path=output_path)
    inputPath = output_path
    i=1
    while( i<3):
        updated_output_path = "D:/color-contrast-analyzer-version_2_2/sample/output/new_gradient/2_"+str(i)+".jpg"
        checkNeighbour(path=inputPath, output_path=updated_output_path)
        i+=1

call_main()