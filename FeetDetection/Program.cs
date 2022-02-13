using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using FeetDetectionLibrary;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;

namespace FeetDetection1
{
    class Program
    {
        static FeetDetection detection = new FeetDetection();
        static FeetDetection.Preprocess preprocess = new FeetDetection.Preprocess();
        /**
         * Process all images from /images/ directory and save results in output directory. 
         */
        static void Main(string[] args)
        {
            Console.WriteLine("Started");
            preprocess.Callibrate("images/perspectiveInput.csv");

            foreach (var dir in Directory.EnumerateDirectories("images"))
            {
                foreach(var filePath in Directory.EnumerateFiles(dir))
                {
                    ProcessAndSaveImage(filePath);
                }
            }
                
            Console.WriteLine("Finished");
        }

        static void ProcessAndSaveImage(string filepath)
        {
            Console.WriteLine("Processing: " + filepath);
            Image<Gray, byte> thresholdImage;
            var image = new Image<Bgr, byte>(filepath);
            image = preprocess.preprocess(image);
            image = image.Resize(720, 480, Inter.Linear);
            var boxes = detection.detectAsPixelCoord(image, out thresholdImage);
            boxes.ForEach(box => image.Draw(box, new Bgr(0, 255, 0), 1));
            var outputPath = GetImageOutputPath(filepath);
            Directory.CreateDirectory("output");
            image.Save(outputPath);
            thresholdImage.Save(GetThresholdOutputPath(filepath));
            Console.WriteLine("Saved: " + outputPath);
        }

        static string GetImageOutputPath(string inputPath)
        {
            var filename = Path.GetFileName(inputPath);
            return Path.Combine("output", filename);
        }

        static string GetThresholdOutputPath(string inputPath)
        {
            var filename = Path.GetFileNameWithoutExtension(inputPath);
            var ext = Path.GetExtension(inputPath);
            return Path.Combine("output", filename + "-thr" + ext);
        }
    }
}