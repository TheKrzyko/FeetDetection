using BenchmarkDotNet.Attributes;
using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceTests
{
    [KeepBenchmarkFiles]
    [CsvExporter]
    [HtmlExporter]
    [MarkdownExporterAttribute.Atlassian]
    public class DetectionBenchmark : PerformanceTestBase
    {
        private List<Image<Bgr, byte>> frames;

        public DetectionBenchmark()
        {
            preprocess.Callibrate("images/perspectiveInput.csv");
        }

        [GlobalSetup]
        public void LoadFrames()
        {
            frames = imageProvider.LoadImagesOfType(imageType)
                .Select(image => preprocess.preprocess(image))
                .Select(image => resizeImage(image))
                .ToList();
        }

        [Benchmark]
        public void Benchmark() => detection.detectAsPixelCoord(frames.PickRandom());

    }
}
