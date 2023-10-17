# Importing modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import extcolors
from colormap import rgb2hex
from numpy import array
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import extcolors
import os
import re
import numpy as np
import extcolors
import os
import re
from collections import Counter

def rgb_to_bgr(rgb):
    # Extract the red, green, and blue components
    red, green, blue = rgb

    # Create the new BGR color value
    bgr = (blue, green, red)

    return bgr


# Define a function to extract the rectangle from the image
def extract_rectangle(image):
    # Convert the image to grayscale
    # Apply OCR to extract the text from the image
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    threshold = 0.1
    rectangles = []  # List to store the extracted rectangles
    text_height_rect = []

    for i, text in enumerate(data['text']):
        if data['conf'][i] > threshold:
            # Filter out non-empty text with confidence scores above the threshold
            if text and len(text.strip()) > 0 and not re.match(r'^\d+$', text):
                # Extract the bounding box coordinates
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                # Add the rectangle coordinates to the list
                rectangles.append((x-2, y-2, x + w + 3, y + h + 2))
                text_height_rect.append(h)

    return rectangles,text_height_rect


def remove_texttes1(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply OCR to extract the text from the image
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    threshold = 10
    for i, text in enumerate(data['text']):
        if data['conf'][i] > threshold:
            # Filter out non-empty text with confidence scores above the threshold
            if text and len(text) > 1 and not re.match(r'^\d+$', text):
                 # Extract the bounding box coordinates
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                if h >= 24:
                    text_large = True
                else:
                    text_large = False
                # Replace the text with gray color
                cv2.rectangle(image, (x-2, y -2), (x + w + 3, y + h + 2), (171, 171, 171), -1)
            
    # cv2.imwrite('text_removed.jpg', image)  # Save the resulting image
    return image


def remove_arrows(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Perform Hough line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90, minLineLength=3, maxLineGap=5)

    # Remove the detected lines from the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (223, 243, 254), thickness=3)  # Draw black lines over the detected lines

    # Apply contour detection on the modified image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find arrowheads among the contours
    arrowheads = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 3:  # Check if the contour is a triangle (potential arrowhead)
            arrowheads.append(approx)

    # Draw the arrowheads on the image
    for arrowhead in arrowheads:
        cv2.drawContours(image, [arrowhead], -1, (32, 255, 42), thickness=30)  # Draw green arrowheads

    return image


def replace_neigh_color_with_original_color_based_on_thres(unique_lst_tuples,unique_rgb_tuples):
    threshold = 20
    for i in range(len(unique_lst_tuples)):
        if len(unique_lst_tuples) >2:
            r1, g1, b1 = unique_lst_tuples[i]
            for j in range(len(unique_rgb_tuples)):
                r2, g2, b2 = unique_rgb_tuples[j]
                if abs(r1 - r2) <= threshold and abs(g1 - g2) <= threshold and abs(b1 - b2) <= threshold:
                    unique_lst_tuples[i] = unique_rgb_tuples[j]
                    break
        else:
            pass

    return list(set(unique_lst_tuples))

def bordercheck(image,for_color,back_color,parent_directory):
    try:
        image = remove_texttes1(image)
    except:
        image = image

    try:
        image = remove_arrows(image)
    except:
        image = image

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)# Convert the image from BGR to HSV color space
    target_colors = [np.uint8([[for_color]]), np.uint8([[back_color]])]  # Example: green and blue colors
    target_colors_hsv = [cv2.cvtColor(color, cv2.COLOR_BGR2HSV) for color in target_colors]# Convert the target colors to HSV color space
    tolerance = 5 # Example: 20 # Set the tolerance for color matching
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8) # Create an empty mask to store the combined color bounds

    for i,color_hsv in zip(range(2),target_colors_hsv):     # Iterate over each target color
        # Get the lower and upper bounds for the color based on the tolerance
        lower_bound = np.array([color_hsv[0][0][0] - tolerance, color_hsv[0][0][1] - tolerance, color_hsv[0][0][2] - tolerance])
        upper_bound = np.array([color_hsv[0][0][0] + tolerance, color_hsv[0][0][1] + tolerance, color_hsv[0][0][2] + tolerance])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)# Create a mask for the current color within the specified bounds
        combined_mask = cv2.bitwise_or(combined_mask, mask)# Add the current color's mask to the combined mask

    kernel = np.ones((3, 3), np.uint8)# Define the structuring element for morphological operations
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)# Perform morphological dilation to close small gaps between objects
    filled_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)# Perform morphological closing to fill larger gaps between objects
    removed_area_image = cv2.bitwise_and(filled_mask, filled_mask, mask=cv2.bitwise_not(combined_mask))# Remove the area of the combined mask from the original image
    
    removed_area_image_path = r'temper34r43t3/removed_area_image.jpg'
    removed_area_image_path1 = os.path.join(parent_directory,removed_area_image_path)
    
    cv2.imwrite(removed_area_image_path1, removed_area_image) # Save the modified image
    removed_area_image = cv2.imread(removed_area_image_path1, cv2.IMREAD_GRAYSCALE)
    mask = cv2.threshold(removed_area_image, 1, 255, cv2.THRESH_BINARY)[1] # Create a mask from the removed area image where white pixels represent the removed area
    result_image = cv2.bitwise_and(image, image, mask=mask)# Apply the mask to the original image to retain only the white area pixels

    # Define the color to fill the rest of the area
    fill_color = (171,171,171)  # gray color in BGR format

    # Create a mask for the non-white areas in the removed area image
    non_white_mask = cv2.bitwise_not(mask)

    # Fill the non-white areas in the result image with the specified color
    result_image[np.where(non_white_mask)] = fill_color

    bordercheck_result_path = r'temper34r43t3/bordercheck_result.jpg'
    bordercheck_result_path1 = os.path.join(parent_directory,bordercheck_result_path)

    # Save the resulting image
    cv2.imwrite(bordercheck_result_path1, result_image)

    # Provide the file path of the image
    image_path = bordercheck_result_path1

    # Extract colors from the image with a tolerance of 10 and a limit of 20 colors
    colors = extcolors.extract_from_path(image_path, tolerance=20, limit = 20)
    count=0
    for color, samples in colors[0]:
        r, g, b = color
        if r <=50 and g <=50 and b <=50:
            count=count+samples
        if 220 <= r < 255 and 220 <= b < 255 and 220 <= b <255:
            count=count+samples    
    return count


def show(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

#This function will plot two images side by side
def plot_image(image, processed_image, title_1='Original Image', title_2='Processed Image'):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#This function will check if an image contain Background and if it contains, then it remove it
def check_and_remove_outer_color(image_path, border_size=3):
    while True:
        image = Image.open(image_path)
        # Calculate the coordinates for checking the border and cropping the image
        width, height = image.size
        outer_left = border_size
        outer_top = border_size
        outer_right = width - border_size - 1
        outer_bottom = height - border_size - 1
        inner_left = border_size + 1
        inner_top = border_size + 1
        inner_right = width - border_size - 2
        inner_bottom = height - border_size - 2
        
        # Get the color of the pixels at each side of the border
        reference_color = image.getpixel((outer_left, outer_top))  # Top-left corner color
        # Check if any side of the border has a different color
        border_color = True
        for x in range(outer_left, outer_right + 1):
            # Check top and bottom sides
            if (
                image.getpixel((x, outer_top)) != reference_color or  # Top side
                image.getpixel((x, outer_bottom)) != reference_color  # Bottom side
            ):
                border_color = False
                break
        
        if border_color:
            for y in range(inner_top, inner_bottom + 1):
                # Check left and right sides
                if (
                    image.getpixel((outer_left, y)) != reference_color or  # Left side
                    image.getpixel((outer_right, y)) != reference_color  # Right side
                ):
                    border_color = False
                    break
        
        if border_color:
            # Crop the image by removing the border
            crop_box = (inner_left, inner_top, inner_right, inner_bottom)
            cropped_image = image.crop(crop_box)
            cropped_image.save(image_path)  # Overwrite the original image with the cropped version
        else:
            break

# Next Below 7 functions to calculate Contrast Ratio
def _linearize(v):
    if v <= 0.03928:
        return v / 12.92
    else:
        return ((v + 0.055) / 1.055) ** 2.4

def rgb(rgb1, rgb2):
    for r, g, b in (rgb1, rgb2):
        if not 0.0 <= r <= 1.0:
            raise ValueError("r is out of valid range (0.0 - 1.0)")
        if not 0.0 <= g <= 1.0:
            raise ValueError("g is out of valid range (0.0 - 1.0)")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b is out of valid range (0.0 - 1.0)")

    l1 = _relative_luminance(*rgb1)
    l2 = _relative_luminance(*rgb2)

    if l1 > l2:
        return (l1 + 0.05) / (l2 + 0.05)
    else:
        return (l2 + 0.05) / (l1 + 0.05)

def _relative_luminance(r, g, b):
    r = _linearize(r)
    g = _linearize(g)
    b = _linearize(b)

    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _linearize(v):
    if v <= 0.03928:
        return v / 12.92
    else:
        return ((v + 0.055) / 1.055) ** 2.4

def passes_AA(contrast, large=True):
    if large:
        return contrast >= 3.0
    else:
        return contrast >= 4.5
    
def passes_AA_textsize(contrast,text_fontsize1):
    return contrast >= 3.0

def passes_AAA(contrast, large=False):
    if large:
        return contrast >= 4.5
    else:
        return contrast >= 7.0


def translate(value, value_min_range, value_max_range, min_range, max_range):
    value_span = value_max_range - value_min_range
    span = max_range - min_range
    scaled = float(value - value_min_range) / float(value_span)
    return min_range + (scaled * span)

def rgb_as_int(rgb1, rgb2):
    n_rgb1 = tuple([translate(c, 0, 255, 0, 1) for c in rgb1])
    n_rgb2 = tuple([translate(c, 0, 255, 0, 1) for c in rgb2])
    return rgb(n_rgb1, n_rgb2)

# Plot Contour
def plot_contour(cc,image):
    # # Load the image
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image_rgb)
    # Define the contour coordinates
    contours = cc[0]
    # Plot the contours
    for contour in contours:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)
    # Show the plot
    return plt.show()

def is_gradient_color(color, gradient_color):
    color_diff = np.abs(color - gradient_color)
    return np.mean(color_diff)

# Function to check Color is gray color or not 
def is_gray_color(rgb,threshold = 20):
    threshold=threshold
    r, g, b = rgb
    if r>=240 and g>=240 and b>=240:
        return False
    if r<=20 and g<=20 and b<=20:
        return False
    if abs(r - g) <= threshold and abs(g - b) <= threshold and abs(b - r) <= threshold:
        return True
    return False

#Function to remove gray color from color_X which is all detected color in image but only when Gray color is not present 
def remove_gray_color(colors_x):
    colors_x=colors_x
    color_x=[]
    for i in range(0,len(colors_x[0])):
        if is_gray_color(colors_x[0][i][0],threshold = 20)== False:
            color_x.append(((colors_x[0][i][0][0],colors_x[0][i][0][1],colors_x[0][i][0][2]),colors_x[0][i][1]))
        else:
            pass
    return(color_x)

def is_color_white(color, range_value):
    white_lower = (255 - range_value, 255 - range_value, 255 - range_value)
    white_upper = (255 + range_value, 255 + range_value, 255 + range_value)
    if all(white_lower[i] <= color[i] <= white_upper[i] for i in range(len(color))):
        return True
    else:
        return False
    
def count_pixels(image_path, color1, color2):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the colors to numpy arrays for easy comparison
    color1 = np.array(color1)
    color2 = np.array(color2)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate the color differences for each pixel
    diff1 = np.abs(image_rgb - color1)
    diff2 = np.abs(image_rgb - color2)

    # Check if each pixel matches either color1 or color2
    matches1 = np.all(diff1 <= 10, axis=2)
    matches2 = np.all(diff2 <= 10, axis=2)

    # Count the number of pixels that match each color
    count1 = np.sum(matches1)
    count2 = np.sum(matches2)

    return count1, count2

def calculate_font_size(image_width, scale_factor=0.5):
    """
    Calculate the font size based on the image width and a scale factor.
    """
    font_scale = (image_width / 12) * scale_factor
    return font_scale

# Identify all images color , plot contour of Background and Foreground 
def save_all_images(path, df, output_folder_path):
    df = df
    path = path
    print(path)
    print(df)
    file_name = os.path.splitext(os.path.basename(path))[0] # Extract the file name from the path
    folder_path = os.path.join(output_folder_path, file_name)
    os.makedirs(folder_path, exist_ok=True) # Create a folder with the file name
    file_name_csv = folder_path + "/failed_cases.csv"

    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb

    def draw_red_square(image_pil, position, size, colorfillbox, colorborder, thickness):
        draw = ImageDraw.Draw(image_pil)
        draw.rectangle([position, (position[0] + size+20, position[1] + size)], fill=colorfillbox, outline=colorborder, width=thickness)

    def rgb_to_bgr(rgb_color):
        r, g, b = rgb_color
        bgr_color = (b, g, r)
        return bgr_color

    for all_comb in range(0, df.shape[0]):
        Background_contours = []
        hex_color_f = ""
        hex_color_b = ""
        for i in ['Background', 'Foreground']:
            target_color = df[i].tolist()[all_comb]
            contrast = "Contrast_Ratio: " + str(round(df['contrast'][all_comb], 2))
            if i == "Background":
                hex_color_b = rgb_to_hex(target_color)

                target_color_back = rgb_to_bgr(target_color)
                hex_color_b = "Background Color: " + hex_color_b
            else:
                hex_color_f = rgb_to_hex(target_color)
                hex_color_f = "Foreground Color: " + hex_color_f
                target_color_for = rgb_to_bgr(target_color)


            image = cv2.imread(path)
            img = Image.open(path)

            # Get the height and width of the image
            height, width = image.shape[:2]
            # Calculate the new height
            new_height = int(height * 1.3)

            # Create a new white image with the new height and same width
            new_img = np.zeros((new_height, width, 3), np.uint8)
            new_img[:] = (255, 34, 43)

            # Copy the original image to the new image
            y_offset = int((new_height - height) / 2)
            new_img[y_offset:y_offset+height, 0:width] = image

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            color_threshold = 50  # Adjust this value based on your color similarity criteria
            color_diffs = np.abs(image_rgb - target_color)
            color_distances = np.sqrt(np.sum(color_diffs ** 2, axis=2))
            mask = color_distances <= color_threshold
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_contour_area = 50  # Adjust this value based on your requirements
            filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

            for contour in filtered_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if i == 'Background':
                    Background_contours.append(contour)
                if i == 'Foreground':
                    cv2.drawContours(image, [contour], 0, (255, 0, 0), 6 )  # Draw a red boundary

            image = image
            # Calculate the new height and width for the blank image
            new_height = image.shape[0] + 75
            image_width = image.shape[1]
            if image_width >= 400:
                new_width = image_width
            else:
                new_width = 400
            # Create a blank image with the new dimensions
            blank_img = np.zeros((new_height, new_width, 3), np.uint8)
            blank_img[:] = (233, 233, 205)

            # Calculate the x position to center the small image on the blank image
            x_pos = int((new_width - image.shape[1]) / 2)

            # Calculate the y position to place the small image at the top
            y_pos = 75

            # Paste the small image onto the blank image
            blank_img[y_pos:y_pos+image.shape[0], x_pos:x_pos+image.shape[1]] = image
            image = blank_img
            # Get the top middle position of the image
            top_middle_x = image.shape[1] // 2
            top_middle_y = 5  # Adjust the y-coordinate as needed
            font_size = 18
            # Add text to the image
            font = ImageFont.truetype("arial.ttf", size= font_size)  # Replace with the path to your font
            image_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(image_pil)
            text_width, text_height = draw.textsize(hex_color_b, font=font)
            text_x = 2

            draw.text((top_middle_x - text_width/2 - 4, top_middle_y), hex_color_b, font=font, fill=(0, 0, 0),
                      align='left')  # Draw centered text

            # Add second text below first text
            second_text_y = top_middle_y + 22  # Adjust the gap between the texts as needed
            second_text = hex_color_f
            second_text_width, second_text_height = draw.textsize(second_text, font=font)
            second_text_x = top_middle_x - second_text_width/2
            draw.text((top_middle_x - text_width/2 -4, second_text_y), second_text, font=font, fill=(0, 0, 0),
                      align='center')  # Draw second text

            # Add Third text below first text
            Third_text_y = second_text_y + 22  # Adjust the gap between the texts as needed
            Third_text = contrast
            Third_text_width, Third_text_height = draw.textsize(second_text, font=font)
            Third_text_x = top_middle_x - Third_text_width/2
            draw.text((top_middle_x - text_width/2 - 4, Third_text_y), Third_text, font=font, fill=(0, 0, 0),
                      align='center')  # Draw second text

            # Draw red square
            square_size = 20
            square_position_line1 = (top_middle_x + text_width/2 + 30, top_middle_y + (text_height - square_size)/2)
            square_position_line2 = (top_middle_x + text_width/2 + 30, top_middle_y + 22 + (text_height - square_size)/2)

            border_thickness = 4  # Adjust the thickness as desired
            if i == "Foreground":
                draw_red_square(image_pil, square_position_line1, square_size, colorfillbox=target_color_back, colorborder=(0, 0, 0), thickness=border_thickness)
                draw_red_square(image_pil, square_position_line2, square_size, colorfillbox=target_color_for, colorborder=(0, 0, 0), thickness=border_thickness)

            image = np.array(image_pil)

            # Save the image
            save_path = os.path.join(folder_path, f'{file_name}_contour_{all_comb}.jpg')
            cv2.imwrite(save_path, image)

    # Saving image names
    files = os.listdir(folder_path)
    images_name_list = []
    if len(files) == 0:
        pass
    else:
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                images_name_list.append(file)
    df['failed_instance'] = images_name_list
    current_time = str(datetime.now())
    df['Time'] = current_time
    df.to_csv(file_name_csv, index=True)
    return df

# Function for Edge detection using Gaussian Blur and Canny
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Apply Gaussian blur to reduce noise
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)  # Perform Canny edge detection
    return edges

#Extract the color values at the edge positions
def detect_edge_colors(image, edges):
    edge_colors = image[edges != 0]  # Extract the color values at the edge positions
    return edge_colors

# Function for Edge Color count
def edge_colors_count(path):
    # Load an image
    path=path
    image = cv2.imread(path)
    edges = detect_edges(image)  # Detect edges in the image
    edge_colors = detect_edge_colors(image, edges)     # Detect edge colors
    colors=[]
    # Display the detected edge colors
    if len(edge_colors) > 0:
        unique_colors = set(tuple(color) for color in edge_colors)
        for color in unique_colors:
            colors.append(color)
    return colors

# Function to check do we have black of whit color
def count_black_white_colors(colors, threshold):
    count = 0
    for color in colors:
        r, g, b = color
        if (r <= threshold and g <= threshold and b <= threshold) or (r >= 255 - threshold and g >= 255 - threshold and b >= 255 - threshold):
            count += 1
    return count

# Function to detect White color and its neighbour contrast
def white_color_checks(path1,colors_x):
        path=path1
        neighbour_color=[]

        # Load the image
        image_path = path  # Replace with the path to your image
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale

        # Threshold the grayscale image to obtain a binary image
        _, binary_image = cv2.threshold(gray_image, 235, 255, cv2.THRESH_BINARY)

        # Find the coordinates of white pixels within the specified range
        white_pixels = np.where((binary_image >= 235) & (binary_image <= 255))

        white_row=[]
        white_col=[]

        ####print the coordinates of the white pixels
        for y, x in zip(white_pixels[0], white_pixels[1]):
            white_row.append(x)
            white_col.append(y)

        img = Image.open(path)
        neighbour_colors=[]
        centre_color=[]
        for color in range(0,len(white_row)):
            x=white_row[color]
            y=white_col[color]
            for adjacent in range(0,3):
                try:
                    color_left = img.getpixel((y, x-adjacent))[0:3]
                    color_right = img.getpixel((y, x+adjacent))[0:3]
                    color_up=img.getpixel((y-adjacent,x))[0:3]
                    color_down=img.getpixel((y+adjacent,x))[0:3]
                    neighbour_colors.append(color_left)
                    neighbour_colors.append(color_right)
                    neighbour_colors.append(color_up)
                    neighbour_colors.append(color_down)
                except:
                    pass

        lst_tuples = [tuple(arr) for arr in neighbour_colors]
        unique_lst_tuples = list(set(lst_tuples))
        unique_rgb_tuples = list(set(rgb_tuple for rgb_tuple, _ in colors_x[0]))
        common_elements=list(set(unique_rgb_tuples) & set(unique_lst_tuples))
       ##print("\ncommon_elements: ",common_elements)
        # common_elements
        centre_color.append((255,255,255))
        neighbour_color.append(common_elements)
        result_c=[]
        result_n=[]
        result_contrast=[]
        result_AA=[]
        source=[]
        for color in range (0,len(centre_color)):
            cc_color=centre_color[color]
            for n_color in range(0,len(neighbour_color[color])):
                nc_color=neighbour_color[color][n_color]
                if cc_color==nc_color:
                    pass
                else:
                    result_c.append(cc_color)
                    result_n.append(nc_color)
                    result_contrast.append(rgb_as_int(cc_color,nc_color))
                    result_AA.append(passes_AA(rgb_as_int(cc_color,nc_color)))
                    source.append(1)
        result=pd.DataFrame()
        result['Background']=result_c
        result['Foreground']=result_n
        result['contrast']=result_contrast
        result['AA_Result']=result_AA
        result['source']=source
        result=result.sort_values(by=['contrast','source'],ascending=[True,False])
        result['A_B']=result['Background'].astype(str)+"_"+result['Foreground'].astype(str)
        result['B_A']=result['Foreground'].astype(str)+"_"+result['Background'].astype(str)
        new=result[['B_A']]
        new['new']=1
        rr=result.merge(new, left_on="A_B",right_on='B_A',how='left')
        duplicate=rr[rr['new']==1]
        unique=rr[rr['new']!=1]
        duplicate_unique=duplicate.drop_duplicates(subset=['contrast'],keep='first')
        result_1=pd.concat([duplicate_unique,unique],axis=0)[['Background','Foreground','contrast','AA_Result','source']]
        return result_1

def black_color_checks(path1,colors_x):
    neighbour_color=[]
    image = cv2.imread(path1)
    # Extract the RGB channels from the image
    b, g, r = cv2.split(image)

    # Create a mask for black pixels within the range (0, 0) to (10, 10)
    black_mask = (b <= 30) & (g <= 30) & (r <= 30)

    # Find the indices of black pixels within the range
    black_pixels_indices = list(zip(*black_mask.nonzero()))
    black_row=[]
    black_col=[]

    for idx in black_pixels_indices:
        black_row.append(idx[0])
        black_col.append(idx[1])

    img = Image.open(path1)
    neighbour_colors=[]
    centre_color=[]
    for color in range(0,len(black_row)):
        x=black_row[color]
        y=black_col[color]
        for adjacent in range(0,1):
            try:
                color_left = img.getpixel((y, x-adjacent))[0:3]
                color_right = img.getpixel((y, x+adjacent))[0:3]
                color_up=img.getpixel((y-adjacent,x))[0:3]
                color_down=img.getpixel((y+adjacent,x))[0:3]
                neighbour_colors.append(color_left)
                neighbour_colors.append(color_right)
                neighbour_colors.append(color_up)
                neighbour_colors.append(color_down)
            except:
                pass

    colors_x=colors_x
    lst_tuples = [tuple(arr) for arr in neighbour_colors]
    unique_lst_tuples = list(set(lst_tuples))
    unique_rgb_tuples = list(set(rgb_tuple for rgb_tuple, _ in colors_x[0]))
    common_elements=list(set(unique_rgb_tuples) & set(unique_lst_tuples))
    # common_elements
    centre_color.append((0,0,0))
    neighbour_color.append(common_elements)

    result_c=[]
    result_n=[]
    result_contrast=[]
    result_AA=[]
    source=[]
    for color in range (0,len(centre_color)):
        cc_color=centre_color[color]
        for n_color in range(0,len(neighbour_color[color])):
            nc_color=neighbour_color[color][n_color]
            if cc_color==nc_color:
                pass
            else:
                result_c.append(cc_color)
                result_n.append(nc_color)
                result_contrast.append(rgb_as_int(cc_color,nc_color))
                result_AA.append(passes_AA(rgb_as_int(cc_color,nc_color)))
                source.append(1)

    result=pd.DataFrame()
    result['Background']=result_c
    result['Foreground']=result_n
    result['contrast']=result_contrast
    result['AA_Result']=result_AA
    result['source']=source
    result=result.sort_values(by=['contrast','source'],ascending=[True,False])
    result['A_B']=result['Background'].astype(str)+"_"+result['Foreground'].astype(str)
    result['B_A']=result['Foreground'].astype(str)+"_"+result['Background'].astype(str)
    new=result[['B_A']]
    new['new']=1
    rr=result.merge(new, left_on="A_B",right_on='B_A',how='left')
    duplicate=rr[rr['new']==1]
    unique=rr[rr['new']!=1]
    duplicate_unique=duplicate.drop_duplicates(subset=['contrast'],keep='first')
    result_1=pd.concat([duplicate_unique,unique],axis=0)[['Background','Foreground','contrast','AA_Result','source']]
    result_1

    return result_1

def bilateral(image, diameter=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)



def count_tuples(lst):
    tuple_counts = Counter(lst)
    sorted_counts = sorted(tuple_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_dict = {k: v for k, v in sorted_counts}
    return sorted_dict


def croping_img(path,parent_directory):
        crop_height=30
        crop_width=60
        image = cv2.imread(path)
        height, width, _ = image.shape# Get the original width and height of the image
        # Calculate the new width and height by reducing 100 pixels from each dimension
        new_width = width - crop_height
        new_height = height - crop_width
        # Calculate the x and y coordinates for cropping
        x = int((width - new_width) / 2)
        y = int((height - new_height) / 2)
        cropped_image = image[y:y+new_height, x:x+new_width] # Crop the image using the new dimensions and coordinates
        # Save the cropped image
        cropped_img_path = r'temper34r43t3/cropped_image.jpg'
        cropped_image_path = os.path.join(parent_directory,cropped_img_path)
        cv2.imwrite(cropped_image_path, cropped_image)
        return cropped_image_path

def smoothing_fun(path,parent_directory):
    image_path = path # Replace with the path to your image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform adaptive thresholding to create a binary mask for the shadows
    block_size = 255  # Adjust the block size as needed
    constant = 25  # Adjust the constant value as needed
    binary_mask = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, block_size, constant)

    # Perform morphological operations to enhance the binary mask
    kernel = np.ones((5,5), np.uint8)  # Adjust the kernel size as needed
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the binary mask
    image_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2) # Draw contours on the original image
    median_filtered_image = cv2.medianBlur(image.copy(), 5)  # Replace each pixel with the median color of its neighbors within a 4x4 kernel

    median_filtered_image_path = r'temper34r43t3/median_filtered_image.jpg'
    median_filtered_image_path1 = os.path.join(parent_directory,median_filtered_image_path)
    cv2.imwrite(median_filtered_image_path1,median_filtered_image)
    checkNeighbour(median_filtered_image_path1,median_filtered_image_path1)
    return median_filtered_image_path1

def checkNeighbour (path, output_path) :
    color_tolerance=20
    min_contour_area = 400
    neighbouringPixels = []
    filtered_contour=[]

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
        try:
            for coordinate in coordinates:
                x,y = coordinate
                image[x,y] = color_rgb
        except:
            pass
        

    #write the final changes to the image
    cv2.imwrite(output_path, image)


def identify_obj_based_all_dim(path,colors_x,color_thresholds,min_contour_areas,adjacent_range,Gradient_check_threshold,parent_directory):
        # Identify the object based on color and get all dimensions of it .
        image = cv2.imread(path)
        img = Image.open(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        centre_color=[]
        neighbour_color=[]
        filtered_contour=[]
        for input_colors in range (0,len(colors_x[0])):             #here color_x[0] is
        #for input_colors in range (1,2):             #here color_x[0] is
            #print("\n\-------------------------------identify_obj_based_all_dim-----------------------------------\n")
            target_color = colors_x[0][input_colors][0]
            #print("\ntarget_color : ",target_color)
            color_threshold = color_thresholds  # Adjust this value based on your color similarity criteria
            color_diffs = np.abs(image_rgb - target_color)
            color_distances = np.sqrt(np.sum(color_diffs ** 2, axis=2))
            mask = color_distances <= color_threshold
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_contour_area = min_contour_areas # Adjust this value based on your requirements
            filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
            filtered_contour.append(filtered_contours)

            #print("filtered_contour -_-_-_: ",filtered_contours)

            # Lets See Neighbour Colors

            neighbour_colors = check_for_each_Conturin_filtered_conturs(filtered_contours,adjacent_range,img,Gradient_check_threshold,target_color,colors_x,image,parent_directory)

            #print("\nneighbour_colors : ",len(neighbour_colors))
            #print("len(neighbour_colors--------)",neighbour_colors)


            lst_tuples = [tuple(arr) for arr in neighbour_colors]
            #print("\nlst_tuples", (len(lst_tuples)))

            try:
                unique_lst_tuples = list(set(lst_tuples))
                #print("\nunique_lst_tuples", (len(unique_lst_tuples)))

                unique_rgb_tuples = list(set(rgb_tuple for rgb_tuple, _ in colors_x[0]))
                #print("\nunique_rgb_tuples", unique_rgb_tuples)

                unique_lst_tuples = replace_neigh_color_with_original_color_based_on_thres(unique_lst_tuples,unique_rgb_tuples)  # Keep Color which is similar to original color with in some threshold 
                #print("\nunique_lst_tuples __-__---------____: ",len(unique_lst_tuples))

                #print("unique_lst_tuples",unique_lst_tuples)

                if len(unique_lst_tuples) > 0:
                    common_elements=list(set(unique_rgb_tuples) & set(unique_lst_tuples[0]))

                else:
                    common_elements=list(set(unique_rgb_tuples) & set(unique_lst_tuples))

                #print("\ncommon_elements", common_elements)

                centre_color.append(target_color)
                neighbour_color.append(common_elements)
                #print("\ncentre_color :",centre_color)
                #print("\nneighbour_color :",neighbour_color)
                #print("###################################################################################################################")
            except:
                pass
        return centre_color,neighbour_color,filtered_contour,image_rgb,neighbour_colors

def check_for_each_Conturin_filtered_conturs(filtered_contours,adjacent_range,img,Gradient_check_threshold,target_color,colors_x,image,parent_directory):
        neighbour_colors=[]
        for i, contour in enumerate(filtered_contours):
            #find boundary of color
            neigh= []
            for boundary in contour:
                x=boundary[0][0]
                y=boundary[0][1]


                for adjacent in range(0,adjacent_range+1):
                    try:

                        color_left = img.getpixel((y, x-adjacent))[0:3]
                        color_right = img.getpixel((y, x+adjacent))[0:3]
                        color_up=img.getpixel((y-adjacent,x))[0:3]
                        color_down=img.getpixel((y+adjacent,x))[0:3]
                        color_left_10 = img.getpixel((y, x-adjacent+3))[0:3]   # For adjacent Neighbour due to gradient need to check
                        color_right_10 = img.getpixel((y, x+adjacent+3))[0:3]
                        color_up_10=img.getpixel((y-adjacent+3,x))[0:3]
                        color_down_10=img.getpixel((y+adjacent+3,x))[0:3]

                        if count_black_white_colors([color_left],20) > 0:
                            neigh.append(color_left)
                        elif np.median(np.abs(np.array(color_left)-np.array(target_color)))>=Gradient_check_threshold:
                            neigh.append(color_left)

                        if count_black_white_colors([color_right],20)> 0:
                            neigh.append(color_right)
                        elif np.median(np.abs(np.array(color_right)-np.array(target_color)))>=Gradient_check_threshold:
                            neigh.append(color_right)

                        if count_black_white_colors([color_up],20)> 0:
                            neigh.append(color_up)
                        elif np.median(np.abs(np.array(color_up)-np.array(target_color)))>=Gradient_check_threshold:
                            neigh.append(color_up)

                        if count_black_white_colors([color_down],20)> 0:
                            neigh.append(color_down)
                        elif np.median(np.abs(np.array(color_down)-np.array(target_color)))>=Gradient_check_threshold:
                            #neighbour_colors.append(color_down)
                            neigh.append(color_down)

                        
                    except:
                        pass

            neigh_tupple = count_tuples(neigh) # count all color 
            tolerance = 20
            colors_to_remove = []

            for color in list(neigh_tupple.keys()):
                if all(abs(color[i] - target_color[i]) <= tolerance for i in range(3)):
                    colors_to_remove.append(color)

            for color in colors_to_remove:
                neigh_tupple.pop(color)

            lst_tuples = [tuple(arr) for arr in list(neigh_tupple.keys())]
            unique_lst_tuples = list(set(lst_tuples))

            unique_rgb_tuples = list(set(rgb_tuple for rgb_tuple, _ in colors_x[0]))
            unique_lst_tuples = replace_neigh_color_with_original_color_based_on_thres(unique_lst_tuples,unique_rgb_tuples)

            common_elements=list(set(unique_rgb_tuples) & set(unique_lst_tuples))
            rgb_back_color = target_color            #RGB color
            if len(common_elements) == 0:
                pass
            else:
                rgb_for_color =  common_elements[0]      #RGB color
                for_color = rgb_to_bgr(rgb_for_color)
                back_color = rgb_to_bgr(rgb_back_color)
                count = bordercheck(image,for_color,back_color,parent_directory)
                neighbour_colors.append(neigh)

        return neighbour_colors

def checking_uniqueness(final_table,gray_color,path_1):
    # # Check Unique rows
    result=final_table.sort_values(by=['contrast','source'],ascending=[True,False])
    result['A_B']=result['Background'].astype(str)+"_"+result['Foreground'].astype(str)
    result['B_A']=result['Foreground'].astype(str)+"_"+result['Background'].astype(str)
    new=result[['B_A']]
    new['new']=1
    rr=result.merge(new, left_on="A_B",right_on='B_A',how='left')
    duplicate=rr[rr['new']==1]
    unique=rr[rr['new']!=1]
    duplicate_unique=duplicate.drop_duplicates(subset=['contrast'],keep='first')
    result_1=pd.concat([duplicate_unique,unique],axis=0)[['Background','Foreground','contrast','AA_Result','source']]

    # Dropping Gray Comparison cases
    if gray_color==True:
        result_2=result_1[['Background',"Foreground","contrast","AA_Result","source"]].drop_duplicates().reset_index(drop=True)
    else:
             # Define a function to check if two colors are almost identical
        def are_colors_almost_identical(color1, color2, threshold=30):
            # Calculate the color difference
            diff = np.abs(np.mean(color1[:3]) - np.mean(color2[:3]))
            return diff <= threshold

        unique_foreground_colors = []
        for index, row in result_1.iterrows():
            is_unique = True
            for color in unique_foreground_colors:
                if are_colors_almost_identical(row['Foreground'], color):
                    is_unique = False
                    break
            if is_unique:
                unique_foreground_colors.append(row['Foreground'])

        result_1_unique = result_1[result_1['Foreground'].apply(lambda x: x in unique_foreground_colors)].reset_index(drop=True)

        result_2 = result_1_unique[['Background', "Foreground", "contrast", "AA_Result", "source"]].drop_duplicates()

    # Continue with the color dominance and filtering logic as before
    result_2 = result_2.reset_index(drop=True)
    count = result_2.shape[0]
    index1 = []
    for i in range(1, count):
        if abs(((result_2['Foreground'].iloc[i][0] + result_2['Foreground'].iloc[i][1] + result_2['Foreground'].iloc[i][2]) / 3) -
               ((result_2['Background'].iloc[i][0] + result_2['Background'].iloc[i][1] + result_2['Background'].iloc[i][2]) / 3)) <= 30:
            index1.append(i)

    result_2.drop(index=index1, inplace=True)
    # Save the failed cases images
    df = result_2[(result_2["AA_Result"] == 0) | (result_2["AA_Result"] == False)].reset_index(drop=True)

    #     result_1['Background_gray'] = result_1['Background'].apply(is_gray_color,threshold = 20)
    #     result_1['Foreground_gray'] = result_1['Foreground'].apply(is_gray_color,threshold = 20)
    #     result_1=result_1[(result_1['Background_gray']!=True)].reset_index(drop=True)
    #     result_1=result_1[(result_1['Foreground_gray']!=True)].reset_index(drop=True)
    #     result_2=result_1[['Background',"Foreground","contrast","AA_Result","source"]].drop_duplicates()
    # # To avoid Gradient and color
    result_2=result_2.reset_index(drop=True)
    count=result_2.shape[0]
    index1=[]
    for i in range(1,count):
        if abs(((result_2['Foreground'].iloc[i][0]+result_2['Foreground'].iloc[i][1]+result_2['Foreground'].iloc[i][2])/3)-\
            ((result_2['Background'].iloc[i][0]+result_2['Background'].iloc[i][1]+result_2['Background'].iloc[i][2])/3))<=30:
            index1.append(i)

    result_2.drop(index=index1, inplace=True)
    # Save the failed cases images
    df=result_2[(result_2["AA_Result"]==0)|(result_2["AA_Result"]==False)].reset_index(drop=True)

    # Check Gradient in Foreground  and Background
    try:
        back=[]
        fore=[]
        for color in df['Background'].unique():
            df1=df[df['Background']==color]
            if df1.shape[0]>1:
                numbers =df1['Foreground'].to_dict()
                lst = []
                for key1, value1 in numbers.items():
                    for key2, value2 in numbers.items():
                        if key1 != key2 and np.median(np.abs(np.array(value1) - np.array(value2))) >20:
                            lst.append([value1, value2])
                unique_elements = list(set(tuple(element) for sublist in lst for element in sublist))
                if len(unique_elements)==0:
                    unique_elements.append(list(numbers.values())[0])

                for i in unique_elements:
                    back.append((df1['Background']).unique()[0])
                    fore.append(i)
            else:
                back.append((df1['Background']).unique()[0])
                fore.append((df1['Foreground']).unique()[0])

        data=pd.DataFrame()
        data['Background']=back
        data['Foreground']=fore
        df=df.merge(data,on=['Background','Foreground'],how='inner')

    except:
        pass

    try:
        back=[]
        fore=[]
        for color in df['Foreground'].unique():
            df1=df[df['Foreground']==color]
            if df1.shape[0]>1:
                numbers =df1['Background'].to_dict()
                lst = []
                for key1, value1 in numbers.items():
                    for key2, value2 in numbers.items():
                        if key1 != key2 and np.median(np.abs(np.array(value1) - np.array(value2))) >20:
                            lst.append([value1, value2])
                unique_elements = list(set(tuple(element) for sublist in lst for element in sublist))
                if len(unique_elements)==0:
                    unique_elements.append(list(numbers.values())[0])
                for i in unique_elements:
                    fore.append((df1['Foreground']).unique()[0])
                    back.append(i)
            else:
                fore.append((df1['Foreground']).unique()[0])
                back.append((df1['Background']).unique()[0])

        data=pd.DataFrame()
        data['Background']=back
        data['Foreground']=fore

        df=df.merge(data,on=['Background','Foreground'],how='inner')
        # Create new columns for counts
        df['background_pixelCount'] = 0
        df['foreground_pixelCount'] = 0

        # Iterate over each row and calculate pixel counts
        for index, row in df.iterrows():
            background_pixelCount, foreground_pixelCount = count_pixels(path_1, row['Background'], row['Foreground'])
            df.at[index, 'background_pixelCount'] = background_pixelCount
            df.at[index, 'foreground_pixelCount'] = foreground_pixelCount

        # Determine the color with the higher pixel count and assign it to the Background column
        df['Background1'] = df.apply(lambda x: x['Background'] if x['background_pixelCount'] >= x['foreground_pixelCount'] else x['Foreground'], axis=1)

        # Determine the color with the lower pixel count and assign it to the Foreground column
        df['Foreground1'] = df.apply(lambda x: x['Foreground'] if x['background_pixelCount'] >= x['foreground_pixelCount'] else x['Background'], axis=1)
        # Drop the unnecessary columns
        df = df.drop(['Background', 'Foreground'], axis=1)
        df.rename(columns={'Background1': 'Background', 'Foreground1':'Foreground'}, inplace=True)
        # Move the renamed column to the first position
        columns = df.columns.tolist()
        columns.insert(0, columns.pop(columns.index('Background')))
        columns.insert(1, columns.pop(columns.index('Foreground')))

        df = df[columns]

        df = df[df['foreground_pixelCount'] >= 500]
        df = df.drop(['background_pixelCount', 'foreground_pixelCount'], axis=1)
        df = df.reset_index(drop=True)
        ######################################################################version 2.2 ends
    except:
        pass


    return df
#This function will be removed
def checking_uniqueness_text(df_text_check,gray_color,path_1):
    # # Check Unique rows
    try:
        result=df_text_check.sort_values(by=['contrast'],ascending=[True,False])
    except:
        result = df_text_check
    result['A_B']=result['Background'].astype(str)+"_"+result['Foreground'].astype(str)
    result['B_A']=result['Foreground'].astype(str)+"_"+result['Background'].astype(str)
    new=result[['B_A']]
    new['new']=1
    rr=result.merge(new, left_on="A_B",right_on='B_A',how='left')
    duplicate=rr[rr['new']==1]
    unique=rr[rr['new']!=1]

    duplicate_unique=duplicate.drop_duplicates(subset=['contrast'],keep='first')
    result_1=pd.concat([duplicate_unique,unique],axis=0)[['Background','Foreground','contrast','AA_Result']]
    # Dropping Gray Comparison cases
    if gray_color==True:
        result_2=result_1[['Background',"Foreground","contrast","AA_Result"]].drop_duplicates().reset_index(drop=True)
    else:
        result_1['Background_gray'] = result_1['Background'].apply(is_gray_color,threshold = 20)
        result_1['Foreground_gray'] = result_1['Foreground'].apply(is_gray_color,threshold = 20)
        result_1=result_1[(result_1['Background_gray']!=True)].reset_index(drop=True)
        result_1=result_1[(result_1['Foreground_gray']!=True)].reset_index(drop=True)
        result_2=result_1[['Background',"Foreground","contrast","AA_Result"]].drop_duplicates()
    # To avoid Gradient and color
    result_2=result_2.reset_index(drop=True)
    count=result_2.shape[0]
    index1=[]
    for i in range(1,count):
        if abs(((result_2['Foreground'].iloc[i][0]+result_2['Foreground'].iloc[i][1]+result_2['Foreground'].iloc[i][2])/3)-\
            ((result_2['Background'].iloc[i][0]+result_2['Background'].iloc[i][1]+result_2['Background'].iloc[i][2])/3))<=30:
            index1.append(i)
    result_2.drop(index=index1, inplace=True)
    # Save the failed cases images
    df=result_2[(result_2["AA_Result"]==0)|(result_2["AA_Result"]==False)].reset_index(drop=True)
    # Check Gradient in Foreground  and Background
    try:
        back=[]
        fore=[]
        for color in df['Background'].unique():
            df1=df[df['Background']==color]
            if df1.shape[0]>1:
                numbers =df1['Foreground'].to_dict()
                lst = []
                for key1, value1 in numbers.items():
                    for key2, value2 in numbers.items():
                        if key1 != key2 and np.median(np.abs(np.array(value1) - np.array(value2))) >20:
                            lst.append([value1, value2])
                unique_elements = list(set(tuple(element) for sublist in lst for element in sublist))
                if len(unique_elements)==0:
                    unique_elements.append(list(numbers.values())[0])

                for i in unique_elements:
                    back.append((df1['Background']).unique()[0])
                    fore.append(i)
            else:
                back.append((df1['Background']).unique()[0])
                fore.append((df1['Foreground']).unique()[0])

        data=pd.DataFrame()
        data['Background']=back
        data['Foreground']=fore
        df=df.merge(data,on=['Background','Foreground'],how='inner')
       ##print("\n3 : ",df)

    except:
        pass

    try:
        back=[]
        fore=[]
        for color in df['Foreground'].unique():
            df1=df[df['Foreground']==color]
            if df1.shape[0]>1:
                numbers =df1['Background'].to_dict()
                lst = []
                for key1, value1 in numbers.items():
                    for key2, value2 in numbers.items():
                        if key1 != key2 and np.median(np.abs(np.array(value1) - np.array(value2))) >20:
                            lst.append([value1, value2])
                unique_elements = list(set(tuple(element) for sublist in lst for element in sublist))
                if len(unique_elements)==0:
                    unique_elements.append(list(numbers.values())[0])
                for i in unique_elements:
                    fore.append((df1['Foreground']).unique()[0])
                    back.append(i)
            else:
                fore.append((df1['Foreground']).unique()[0])
                back.append((df1['Background']).unique()[0])

        data=pd.DataFrame()
        data['Background']=back
        data['Foreground']=fore

        df=df.merge(data,on=['Background','Foreground'],how='inner')
        # Create new columns for counts
        df['background_pixelCount'] = 0
        df['foreground_pixelCount'] = 0

        # Iterate over each row and calculate pixel counts
        for index, row in df.iterrows():
            background_pixelCount, foreground_pixelCount = count_pixels(path_1, row['Background'], row['Foreground'])
            df.at[index, 'background_pixelCount'] = background_pixelCount
            df.at[index, 'foreground_pixelCount'] = foreground_pixelCount

        # Determine the color with the higher pixel count and assign it to the Background column
        df['Background1'] = df.apply(lambda x: x['Background'] if x['background_pixelCount'] >= x['foreground_pixelCount'] else x['Foreground'], axis=1)

        # Determine the color with the lower pixel count and assign it to the Foreground column
        df['Foreground1'] = df.apply(lambda x: x['Foreground'] if x['background_pixelCount'] >= x['foreground_pixelCount'] else x['Background'], axis=1)
        # Drop the unnecessary columns
        df = df.drop(['Background', 'Foreground'], axis=1)
        df.rename(columns={'Background1': 'Background', 'Foreground1':'Foreground'}, inplace=True)
        # Move the renamed column to the first position
        columns = df.columns.tolist()
        columns.insert(0, columns.pop(columns.index('Background')))
        columns.insert(1, columns.pop(columns.index('Foreground')))

        df = df[columns]

        df = df[df['foreground_pixelCount'] >= 500]
        df = df.drop(['background_pixelCount', 'foreground_pixelCount'], axis=1)
        df = df.reset_index(drop=True)
        ######################################################################version 2.2 ends
    except:
        pass


    return df



def remove_gray_color_from_color_x(colors_x):

    if is_gray_color(colors_x[0][0][0])==True or is_gray_color(colors_x[0][1][0])==True or \
            is_gray_color(colors_x[0][2][0])==True:
        gray_color=True
    else:
        gray_color=False

    if gray_color==False:
        colors_x=(remove_gray_color(colors_x),colors_x[1])      # To remove gray color which can be shadow

    return gray_color,colors_x

## Function to check color contrast
def color_contrast_check(path_1,output_folder_path,parent_directory):
    print("before size: ",os.path.getsize(path_1))

    #adding code for extracting image contur e version 2.2.1
    image = cv2.imread(path_1)

    #________version 2.3 starts______-------------------------------
    # Call the function with your image
    rectangles,text_height_rect = extract_rectangle(image)

    df_text_check = pd.DataFrame(columns =['Background','Foreground','contrast','AA_Result','source'])

    # Extract the rectangles from the image
    for rect,rectnum,height_txt in zip(rectangles,range(len(rectangles)),text_height_rect):
        try:
            x1, y1, x2, y2 = rect
            extracted_rectangle = image[y1:y2, x1:x2]

            # Do something with the extracted rectangle, e.g., save it
            cv2.imwrite('{}extracted_rectangle.png'.format(rectnum), extracted_rectangle)

            colors_x = extcolors.extract_from_path('{}extracted_rectangle.png'.format(rectnum), tolerance =30, limit = 3)
            # delete the file
            os.remove('{}extracted_rectangle.png'.format(rectnum))
            background_data = colors_x[0][0][0]
            foreground_data = colors_x[0][1][0]
            contrast_data = rgb_as_int(background_data,foreground_data)
            aa_result_data = passes_AA_textsize(contrast_data,height_txt)
                # Add the data to the DataFrame
            df_text_check.loc[len(df_text_check)] = [background_data, foreground_data, contrast_data, aa_result_data,2]

        except:
            pass

    df_text_check = df_text_check.drop_duplicates()
    df_text_check = df_text_check.reset_index(drop=True)
    df_text_check = df_text_check[df_text_check['AA_Result'] == False]
    df_text_check = df_text_check.reset_index(drop=True)
    gray_color = False

    df_text_check = checking_uniqueness(df_text_check,gray_color,path_1)

    #-----------------___________------------ version 2.3 ends________________------------------------

    #preprocessing
    path=path_1

    gray_color=False
    crop_required=True
    color_tolerance=15
    color_thresholds=20  # Adjust this value based on your color similarity criteria This variable determines the threshold for color similarity. Adjusting this value will affect which colors are considered similar.
    min_contour_areas=200 # Adjust this value based on your requirements
    Gradient_check_threshold=40
    smoothing=False
    adjacent_range=2

    # Smoothing Required or not
    colo=edge_colors_count(path)

    if len(set(colo))>=5000: smoothing=True

    # Deciding What should be value of adjacent Range
    count = count_black_white_colors(colo, 20)

    #adjacent_range to check if need to check for 1 px, 2px or 3px
    adjacent_range = [3 if count >= 0 and count <= 100 else 2\
                      if count > 100 and count <= 400 else 1][0]

    # Cropping of image
    if crop_required==True:
        path = croping_img(path,parent_directory)


    # Calculate Color in images
    adjacent_range=adjacent_range  # Keep 1 if border in image
    Gradient_check_threshold=Gradient_check_threshold #for boundary images higher means greater than 25
    if(os.path.getsize(path)>200000):
        output_width = 900                  #set the output size
        img = Image.open(path)
        wpercent = (output_width/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((output_width,hsize), Image.ANTIALIAS)
        # resize_name = 'static/temp/resize_photo.png'  #the resized image name
        resize_name_path = r'temper34r43t3/resize_photo.png'
        resize_name = os.path.join(parent_directory,resize_name_path)

        img.save(resize_name)

        plt.figure(figsize=(3,3))
        img_url = resize_name
        img = plt.imread(img_url)
        print("after size: ",os.path.getsize(img_url))
    else:
        output_width = 900                 #set the output size
        img = Image.open(path)
        wpercent = (output_width/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((output_width,hsize), Image.ANTIALIAS)
        # resize_name = 'static/temp/resize_photo.png'  #the resized image name
        resize_name_path = r'temper34r43t3/resize_photo.png'
        resize_name = os.path.join(parent_directory,resize_name_path)

        img.save(resize_name)

        plt.figure(figsize=(3,3))
        img_url = resize_name
        img = plt.imread(img_url)
        print("after size: ",os.path.getsize(img_url))


    #findings all colors in image
    
    colors_x = extcolors.extract_from_path(img_url, tolerance = color_tolerance, limit = 30)
    color_tolerance = 1 if colors_x[1] <= 100000 else 5 if 100000 < colors_x[1] <= 300000 else \
                    10 if 300000 < colors_x[1] <= 500000 else 15

    #Logic for color tolerance
    k=100*(colors_x[0][0][1]+colors_x[0][2][1]+colors_x[0][3][1])/colors_x[1]
    if k<=80 and k>60:
        colors_x = extcolors.extract_from_path(img_url, tolerance = color_tolerance, limit = 30)
        Gradient_check_threshold=50
    if k<=55:
        colors_x = extcolors.extract_from_path(img_url, tolerance =color_tolerance, limit = 30)
        Gradient_check_threshold=60

    # removing colors which having low pixel count and can be gradient color and below code is to speed up process 
    #update( version 2.1)#######

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

    # To remove Gray Color from Color_x list because it creates distortion

    gray_color,colors_x = remove_gray_color_from_color_x(colors_x)

    # Code to run White color checks in image , it will trigger based on logic
    total_count = colors_x[1]
    target_color = (255, 255, 255)
    threshold = 10
    count = 0
    for color_tuple in colors_x[0]:
        color, color_count = color_tuple
        # Calculate the absolute difference for each channel
        diff = max(abs(color[0] - target_color[0]), abs(color[1] - target_color[1]), abs(color[2] - target_color[2]))
        # Check if the difference is within the threshold
        if diff <= threshold:
            count += color_count

    # Calculate the relative count
    relative_count = count / total_count

    if relative_count>= 0.5 and (colors_x[0][0][0]==(255,255,255) or colors_x[0][1][0]==(255,255,255)):
        white_color_check =white_color_checks(img_url,colors_x)

    else:
        white_color_check=pd.DataFrame(columns=['Background', 'Foreground', 'contrast', 'AA_Result', 'source'])

  # Code to run black color chec
    black_color_check=black_color_checks(img_url,colors_x)

    if smoothing==True:
        path = smoothing_fun(img_url,parent_directory)
    else:
        path = img_url

    centre_color,neighbour_color,filtered_contour,image_rgb,neighbour_colors = identify_obj_based_all_dim(path,colors_x,color_thresholds,min_contour_areas,adjacent_range,Gradient_check_threshold,parent_directory)

  # Create dataframe
    result_c=[]
    result_n=[]
    result_contrast=[]
    result_AA=[]
    source=[]

    for color in range (0,len(centre_color)):
        cc_color=centre_color[color]
        for n_color in range(0,len(neighbour_color[color])):
            nc_color=neighbour_color[color][n_color]
            if cc_color==nc_color:
                pass
            else:
                result_c.append(cc_color)
                result_n.append(nc_color)

                result_contrast.append(rgb_as_int(cc_color,nc_color))
                result_AA.append(passes_AA(rgb_as_int(cc_color,nc_color)))
                source.append(1)

    result=pd.DataFrame()
    result['Background']=result_c
    result['Foreground']=result_n
    result['contrast']=result_contrast
    result['AA_Result']=result_AA
    result['source']=source

    final_table=pd.concat([white_color_check,black_color_check,result],axis=0).drop_duplicates()
    # Merging the data frames df_text_check into final_table
    merged_df = pd.concat([df_text_check, final_table], ignore_index=True)

    df = checking_uniqueness(merged_df,gray_color,path_1)
    
    df_failedcases = save_all_images(path_1,df,output_folder_path)

    print(path_1,color_tolerance,',adjacent_range: ',adjacent_range,'\nGradient_check_threshold :',Gradient_check_threshold,',smoothing: ',smoothing,',gray_color :',gray_color,"\n")
    return df,all_colors,final_table,df_failedcases                 #used all_colors to remove gradients from ignored colors image

#Function to plot the avoided points
def check_color(all_color,image_path,output_path,df,image_rgb):
    color=[]
    for i in range(0,len(all_color[0])):
        color.append(all_color[0][i][0])
    checked=list(set(df['Background'].to_list()+df['Foreground'].to_list()))
    not_identified =list(set(checked)-set(color))
    # Iterate over each color in the "not_identified" list
    for color in not_identified:
        # Create a mask for the current color
        lower_color = np.array(color, dtype=np.uint8)
        upper_color = np.array(color, dtype=np.uint8)
        mask = cv2.inRange(image_rgb, lower_color, upper_color)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the image with green color
        cv2.drawContours(image_rgb, contours, -1, (0, 255, 0), 5)
    # Save the output image
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
