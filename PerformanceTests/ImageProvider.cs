using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceTests
{
    public class ImageProvider
    {
        public IEnumerable<Image<Bgr, byte>> LoadImagesOfType(string type)
        {
            return Directory.EnumerateFiles("images/"+type)
                .Select(filePath => new Image<Bgr, byte>(filePath));
        }
    }
}
