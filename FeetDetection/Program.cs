using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using FeetDetectionLibrary;
using System;

namespace FeetDetection1
{
    class Program
    {
        static void Main(string[] args)
        {
            var detection = new FeetDetection();
            var file = "IMG_3522.JPG";
            var image = detection.LoadFromFile(file);
            //var img2 = image.Resize(900, 600, Inter.Linear);
            var boxes = detection.detect(image);
            foreach (var box in boxes)
            {
                image.Draw(new Ellipse(box), new Bgr(0, 0, 1));
            }
            CvInvoke.Imshow("Window", image);
            CvInvoke.WaitKey();
        }
    }
}