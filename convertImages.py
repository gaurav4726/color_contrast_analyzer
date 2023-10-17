import subprocess


def convert_images(svgFilePath, outputPath):
    javascript_validation_script = "img.js"

    # svgFilePath = "D:/color-contrast-analyzer-data/SVGs/SVGs/m180_msp_11_b2t2l5_001.svg"
    # outputPath = "D:/color-contrast-analyzer-data/SVGs/jpgs/m180_msp_11_b2t2l5_001-1.jpeg"
    js_script_command = ["node", javascript_validation_script,
                         svgFilePath, outputPath]

    try:
        output = subprocess.check_output(
            js_script_command, stderr=subprocess.STDOUT, text=True)
        print("Javascript output: ", output)
    except subprocess.CalledProcessError as e:
        print("Error: ", e)


# convert_images
#("D:/color-contrast-analyzer-data/resize-image/input")
