using BenchmarkDotNet.Attributes;
using Emgu.CV;
using Emgu.CV.Structure;
using FeetDetectionLibrary;
using Microsoft.Diagnostics.Tracing.StackSources;
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
    public class FixPerspectiveBenchmark: PerformanceTestBase
    {
        private List<Image<Bgr, byte>> frames;

        public FixPerspectiveBenchmark()
        {
            preprocess.Callibrate("images/perspectiveInput.csv");
        }

        [GlobalSetup]
        public void LoadFrames()
        {
            frames = imageProvider.LoadImagesOfType(imageType)
                .Select(image => resizeImage(image))
                .ToList();
        }

        [Benchmark]
        public void Benchmark() => preprocess.preprocess(frames.PickRandom());

        
    }
}
