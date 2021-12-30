using BenchmarkDotNet.Attributes;
using Emgu.CV;
using Emgu.CV.Structure;
using FeetDetectionLibrary;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceTests
{
    public abstract class PerformanceTestBase
    {
        protected readonly FeetDetection.Preprocess preprocess = new FeetDetection.Preprocess();
        protected readonly FeetDetection detection = new FeetDetection();
        protected readonly ImageProvider imageProvider = new ImageProvider();

        public IEnumerable<string> ImageTypes => new[] { "clean", "partly-noisy", "noisy" };
        public IEnumerable<Resolution> Resolutions => Resolution.ListAllDefined();

        [ParamsSource(nameof(ImageTypes))]
        public string imageType;

        [ParamsSource(nameof(Resolutions))]
        public Resolution resolution;

        protected Image<Bgr, byte> resizeImage(Image<Bgr, byte> image)
        {
            return resolution.IsNative ? image : image.Resize(resolution.Width, resolution.Height, Emgu.CV.CvEnum.Inter.Linear);
        }
    }
}
