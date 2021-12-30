using BenchmarkDotNet.Running;
using System;

namespace PerformanceTests
{
    class Program
    {
        static void Main(string[] args)
        {
            BenchmarkRunner.Run<FixPerspectiveBenchmark>();
            BenchmarkRunner.Run<DetectionBenchmark>();
            BenchmarkRunner.Run<FullProcessingBenchmark>();
        }
    }
}
