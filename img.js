const svgToImg = require("svg-to-img");
const fs = require("fs");
const sharp = require("sharp");

const convert_SVG_to_JPEG = async (svgFilePath, outputPath) => {
  //   const svgFilePath =
  //     "D:/color-contrast-analyzer-data/SVGs/SVGs/m180_msp_11_b2t2l5_001.svg";
  //   const outputPath =
  //     "D:/color-contrast-analyzer-data/SVGs/jpgs/m180_msp_11_b2t2l5_001.jpeg";

  // Read the SVG file content from the file
  try {
    const svgContent = await fs.promises.readFile(svgFilePath, "utf-8");
    const jpegBuffer = await svgToImg.from(svgContent).toJpeg();
    const tempImgPath = "temp.jpeg";
    // await fs.promises.writeFile(tempImgPath, jpegBuffer);
    // await sharp(tempImgPath)
    //   .resize({ width: 300, height: 300, fit: "inside" })
    //   .toFile(jpegPath);

    // await fs.promises.unlink(tempImgPath);
    // Convert SVG content to JPEG
    await svgToImg.from(svgContent).toJpeg({
      path: outputPath,
    });
    console.log("SVG converted to JPEG successfully.");
  } catch (error) {
    console.error("Error converting SVG to JPEG:", error);
  }
};

const svgPath = process.argv[2];
const outputPath = process.argv[3];
convert_SVG_to_JPEG(svgPath, outputPath);
