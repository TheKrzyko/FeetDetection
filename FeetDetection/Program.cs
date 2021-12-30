using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using FeetDetectionLibrary;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;

namespace FeetDetection1
{
    class Program
    {
        private static FeetDetection detection = new FeetDetection();
        static void Main(string[] args)
        {
            var csv = new StringBuilder();
            for(int i = 0; i < 10; i++)
            {
                DetectVideo("perspectiveInput.csv");
            }
            
            var file = "IMG_3522.JPG";
            var image = detection.LoadFromFile(file);
            var img2 = image.Resize(900, 600, Inter.Linear);
            var boxes = detection.detect(img2);
            foreach (var box in boxes)
            {
                img2.Draw(box, new Bgr(0, 255, 0), 1);
                //img2.Draw(new Ellipse(box), new Bgr(0, 0, 255));
            }
            //CvInvoke.Imshow("Window", img2);
            
        }

        static void DetectVideo(string inputPerspectiveFilename)
        {
            var perspective = new FeetDetection.Preprocess();
            perspective.Callibrate(inputPerspectiveFilename);
            var videoFile = "video.mp4";
            var camera = new VideoCapture(videoFile);
            var fourcc = VideoWriter.Fourcc('m', 'p', '4', 'v');
            var frame = new Mat();
            var counter = 0;
            while (camera.IsOpened)
            {
                camera.Read(frame);
                if (frame.IsEmpty)
                    break;
                counter++;
                var img2 = frame.ToImage<Bgr, byte>();
                var img3 = perspective.preprocess(img2).Resize(900, 600, Inter.Linear);
                var boxes = detection.detect(img3);

                //videoWriter.Write(img2);
            }
            //videoWriter.Dispose();
        }
    }
}