using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using FeetDetectionLibrary;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace FeetDetection1
{
    class LiveFeetDetector: IDisposable
    {
        public bool Life { get; set; }
        public List<RotatedRect> Boxes { get; set; }
        private readonly FeetDetection detection = new FeetDetection();
        private Task detectingJob;
        private FeetDetection.Preprocess perspective;
        public bool IsRunning { get; private set; }
        public LiveFeetDetector(bool Life=true)
        {
            this.Life = Life;
            StartDetecting();
        }

        private void StartDetecting()
        {
            perspective = new FeetDetection.Preprocess();
            perspective.TestCallibrate();
           
            var videoFile = "video.mp4";
            var camera = new VideoCapture(videoFile);
            var fourcc = VideoWriter.Fourcc('m', 'p', '4', 'v');
            var frame = new Mat();

            detectingJob = Task.Run(() => Detect(camera));
            detectingJob.Start();
            IsRunning = true;
            //videoWriter.Dispose();
        }

        public void StopDetecting()
        {
            IsRunning = false;
        }

        private void Detect(VideoCapture camera)
        {
            var frame = new Mat();
            while (camera.IsOpened && IsRunning)
            {
                camera.Read(frame);
                if (frame.IsEmpty)
                    break;
                var img2 = frame.ToImage<Bgr, byte>();
                var img3 = perspective.preprocess(img2).Resize(900, 600, Inter.Linear);
                this.Boxes = detection.detect(img3);
                //videoWriter.Write(img2);
            }
        }

        public void Dispose()
        {
            IsRunning = false;
        }
    }
}