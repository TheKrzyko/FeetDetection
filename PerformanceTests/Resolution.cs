using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceTests
{
    public class Resolution
    {
        public int Width { get; private init; }
        public int Height { get; private init; }

        public bool IsNative => Width == 0 && Height == 0;
        private Resolution(int width, int height)
        {
            Width = width;
            Height = height;
        }

        public static Resolution Native()
        {
            return new Resolution(0, 0);
        }
        public static Resolution R480p()
        {
            return new Resolution(720, 480);
        }
        public static Resolution HD()
        {
            return new Resolution(1280, 720);
        }
        public static Resolution FullHD()
        {
            return new Resolution(1920, 1080);
        }

        public static List<Resolution> ListAllDefined()
        {
            return new List<Resolution>
            {
                Native(),
                FullHD(),
                HD(),
                R480p()
            };
        }

        public override string ToString()
        {
            if (IsNative)
                return "native";
            return Width.ToString() + "x" + Height.ToString();
        }
    }
}
