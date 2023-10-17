from flask import Flask, request, render_template
import glob
import cv2
import os
import logging
# logging.basicConfig(level=logging.INFO)  # Set the desired logging level
import pandas as pd
from app import check_and_remove_outer_color, color_contrast_check, check_color
import extcolors
from datetime import datetime
import pdb
from convertImages import convert_images
import time


app = Flask(__name__)
# Function which will take input of csv file and append all results in it .
def append_to_csv(dataframe, file_path):
    if os.path.exists(file_path):
        # If the file already exists, append the DataFrame to it
        dataframe.to_csv(file_path, mode='a', header=False, index=False)
    else:
        # If the file doesn't exist, create a new CSV file and write the DataFrame to it
        dataframe.to_csv(file_path, index=False)
def get_folder_names(output_folder_path):
    folder_names = []
    sorted_directories = []
    for root, dirs, files in os.walk(output_folder_path):
        if dirs: sorted_directories = sorted(dirs, key=lambda d: get_directory_creation_time(os.path.join(root, d))) # Sort the directories based on time of creation to restart the process and del last created folder
        for dir_name in dirs:
            folder_names.append(dir_name)
    return folder_names, sorted_directories
def get_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']  # Add more extensions if needed
    image_filess = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_filess.append(os.path.splitext(file)[0] )
    
    return image_filess

# Function that get's the creation time of a directory
def get_directory_creation_time(directory_path):
    return os.path.getctime(directory_path)

# Api to run all Function
@app.route('/', methods=['GET','POST'])
def path(input_folder_path):
        start_time=time.time()
    # if request.method == 'POST':
        # input_folder_path = request.form.get('folder-path')
        file_names=get_image_files(input_folder_path)  # New added
        # Get the parent directory of the input folder
        parent_directory = os.path.dirname(input_folder_path)
        # Create a temporary folder in the parent directory
        temp_folder = os.path.join(parent_directory, 'temper34r43t3')
        # print(parent_directory)
        # Check if the temporary folder already exists
        if not os.path.exists(temp_folder):
            # Create the temporary folder if it doesn't exist
            os.makedirs(temp_folder)

        # Name of the output folder
        output_folder_name = 'output'
        # Path to the output folder
        output_folder_path = os.path.join(parent_directory, output_folder_name)

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
        
        # get sorted directories from the get_folder_names function
        folder_list, sorted_directories = get_folder_names(output_folder_path)# New added
        if 'images_ignored_color' in folder_list:folder_list.remove('images_ignored_color')# New added
        if '._output' in folder_list:folder_list.remove('._output')    # New added
        if 'images_ignored_color' in sorted_directories:sorted_directories.remove('images_ignored_color')
        if '._output' in sorted_directories:sorted_directories.remove('._output')
        
        print("sorted by creation time:        ",sorted_directories)
        if sorted_directories: sorted_directories.pop()
        print("after removing the last created folder:", sorted_directories)
        folder_list = sorted_directories

        #logging.info(f"Input Folder Path: {input_folder_path}")
        #logging.info(f"Output Folder Path: {output_folder_path}")

        # Specify the directory path where you want to search for images
        # Use glob to search for image file paths
        images_path = glob.glob(input_folder_path + r'\**\*.png', recursive=True)
        images_path += glob.glob(input_folder_path + r'\**\*.jpg', recursive=True)
        images_path += glob.glob(input_folder_path + r'\**\*.jpeg', recursive=True)
        images_path += glob.glob(input_folder_path + r'\**\*.tiff', recursive=True)
        svg_images_path = glob.glob(input_folder_path + r'\**\*.svg', recursive=True)
        #convert svg file to jpeg and store them in temp folder
        for svg_path in svg_images_path :
            svg_file_name = os.path.splitext(os.path.basename(svg_path))[0]
            file_names.append(svg_file_name)
            svg_file_jpeg_name = svg_file_name+'.jpeg'
            temp_folder_svg_path = os.path.join(temp_folder,svg_file_jpeg_name).replace('/','\\')
            # temp_folder_svg_path = temp_folder+'\\'+svg_file_name+'.jpeg';
            print(temp_folder_svg_path)
            convert_images(svg_path,temp_folder_svg_path)
            images_path.append(temp_folder_svg_path)
        print(images_path)

        final_df = pd.DataFrame() # Final dataframe for all images issue 
        # for i in images_path:
        #     try:
        #         check_and_remove_outer_color(i) 
        #     except:
        #         # print("Got error in reading this image!!")
        #         pass

        # images_path = glob.glob(input_folder_path + r'\**\*.png', recursive=True)
        # images_path += glob.glob(input_folder_path + r'\**\*.jpg', recursive=True)
        # images_path += glob.glob(input_folder_path + r'\**\*.jpeg', recursive=True)

        # New added
        diff=list(set(file_names)-set(folder_list))
        images_path = [path for path in images_path if os.path.splitext(os.path.basename(path))[0] in diff]
        common_elements = list(set(file_names).intersection(folder_list))
        all_images=pd.DataFrame(file_names)
        done_images=pd.DataFrame(common_elements)
        done_images['Status']='Done'
        try:
            all_done_images=all_images.merge(done_images,how='left')
        except:
            all_done_images=all_images
            all_done_images['Status']='New run'

        # creating a new all_done_images.csv to store all the images running status
        all_done_images['Status']=all_done_images['Status'].fillna("New Run")
        all_done_images.columns=['images_name','Status']
        all_done_images_csv_name =os.path.join(parent_directory,"all_done_images.csv")
        # Drop duplicate rows
        all_done_images = all_done_images.drop_duplicates()
        all_done_images.to_csv(all_done_images_csv_name,index=False)
        
        print(all_done_images_csv_name)


        for ijk,j in zip(images_path,range(len(images_path))):
            starttime = datetime.now()
            try:
                processbar_df = pd.DataFrame(columns=['name', 'status', 'start time', 'endtime'])

                t =os.path.splitext(os.path.basename(ijk))[0]
                all_done_images.loc[all_done_images['images_name'] == str(t), 'start-time'] = datetime.now()


                image = cv2.imread(ijk)
                # Get the relative path of the current file
                relative_path = os.path.relpath(os.path.dirname(ijk), input_folder_path)

                # Construct the output folder path
                output_subfolder = os.path.join(output_folder_path,'_output')

                # Create the output subfolder if it doesn't exist
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                df,all_color,final_table, df_failedcases =color_contrast_check(ijk,output_subfolder,parent_directory)
                filenames = os.path.basename(ijk)
                df['source_image'] = str(filenames)
                final_df = pd.concat([final_df, df],axis=0)
                final_df.reset_index(drop=True)


                
                # folder_name = 'images_ignored_color'    # Specify the name of the new folder

                # Join the directory path and folder name
                # output_folder_path1 = os.path.join(output_subfolder, folder_name)

                # Create the new folder if it doesn't exist
                # if not os.path.exists(output_folder_path1):
                #     os.makedirs(output_folder_path1)

                filenames_ignored = 'ignored_colors{}'.format(filenames)
                file_name_withoutext = os.path.splitext(filenames)[0]

                # output_path = os.path.join(output_subfolder, )
                output_pathfinal = os.path.join(output_subfolder,file_name_withoutext,filenames_ignored)

                all_color = extcolors.extract_from_path(ijk, tolerance = 20, limit = 20)

                #print(output_pathfinal)
                check_color(all_color,ijk,output_pathfinal,final_table,image_rgb)
            
                final_df_csv_name =os.path.join(output_folder_path,"final_df.csv")
                # Move a column to index 0
                column_to_move = 'source_image'
                new_columns = [column_to_move] + [col for col in df_failedcases.columns if col != column_to_move]
                df_failedcases = df_failedcases[new_columns]
                append_to_csv(df_failedcases, final_df_csv_name)

                # print(type(processbar_df))
                # print(processbar_df.head())

                processbar_df = pd.concat([processbar_df, pd.DataFrame({'name': [ijk],
                                                                'status': ['done'],
                                                                'start time': [starttime],
                                                                'endtime': [datetime.now()]})],ignore_index=True)

                processbar_df = processbar_df.drop_duplicates()
                all_done_images.loc[all_done_images['images_name'] == str(t), 'Status'] = 'Done' 
                all_done_images.loc[all_done_images['images_name'] == str(t), 'end-time'] = datetime.now()
                all_done_images.to_csv(all_done_images_csv_name,index=False)

                processbar_csv_path = os.path.join(parent_directory,r'processbar.csv')
                processbar_df.to_csv(processbar_csv_path, mode='a', index=False, header=not os.path.isfile(processbar_csv_path))
            except Exception as e:
                print("An exception occurred:", str(e))
                processbar_df = pd.DataFrame(columns=['name', 'status', 'start time', 'endtime'])
                # processbar_df = processbar_df.append({'name': ijk, 
                #                                       'status': 'not done',
                #                                         'start time': starttime,
                #                                           'endtime': datetime.now()}, ignore_index=True)
                processbar_df = pd.concat([processbar_df, pd.DataFrame({'name': [ijk],
                                                             'status': ['not done'],
                                                             'start time': [starttime],
                                                             'endtime': [datetime.now()]})],
                              ignore_index=True)
               
                processbar_df = processbar_df.drop_duplicates()
                processbar_csv_path = os.path.join(parent_directory,r'processbar.csv')
                processbar_df.to_csv(processbar_csv_path, mode='a', index=False, header=not os.path.isfile(processbar_csv_path))
                pass

        #removing temp files
        folder_path = temp_folder
        # Get a list of all files in the folder
        file_list = os.listdir(folder_path)

        # Iterate over the files and remove them one by one
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        
        os.rmdir(temp_folder)

        image_num = len(images_path)
        print('Images processed :', image_num)
        end_time=time.time()
        total_time=end_time-start_time
        print("total time taken " , total_time)

        return output_folder_path,image_num

        # return render_template('colorcontrast.html',output_section=True,output_folder_path = output_folder_path,image_num = len(images_path))

        # return '''<h1>The Input path is : {}</h1>
        #         <h1>The Output path is : {}</h1>'''.format(input_folder_path, output_folder_path)
    # return render_template('colorcontrast.html',output_section=False)

# if __name__ == '__main__':
#     app.run(debug=True, port=8022)


# Need to pass input folder name and output folder name 
#http://127.0.0.1:5000/path?input_folder_path=C:\Users\v-gaurav.gupta\Desktop\color_contrast\Demo\input&output_folder_path=C:\Users\v-gaurav.gupta\Desktop\color_contrast\output