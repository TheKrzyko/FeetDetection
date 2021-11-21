using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Ocl;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace FeetDetectionLibrary
{
    public class FeetDetection
    {
        public class Preprocess
        {
            private bool doPerspective = false;
            private Mat warpMatrix = null;
            public void TestCallibrate()
            {
                int[,] srcv = { {1300, 574},{2834, 302},{1114, 2521},{2596, 3136} };
                Matrix<int> src = new Matrix<int>(srcv);
                int[,] dstv =  {{0,0},{2400,0},{0,2400},{2400,2400}};
                Matrix<int> dst = new Matrix<int>(dstv);
                this.CallibratePerspective(src,dst);

            }
            public void CallibratePerspective(IInputArray src, IInputArray dst)
            { 
                doPerspective = true;
                this.warpMatrix = CvInvoke.GetPerspectiveTransform(src, dst);
            }
            public Image<Bgr, byte> preprocess(Image<Bgr, byte> raw)
            {
                var outSize = new Size(2400, 2400);
                Image<Bgr, byte> src = raw;
                Image<Bgr, byte> dst = new Image<Bgr, byte>(raw.Size);
                if (doPerspective)
                    CvInvoke.WarpPerspective(src, dst, this.warpMatrix, outSize);
                return dst;
            }
        }

        public bool isFootProprtion(float width, float height)
        {
            float minFootProportion = 1.4f;
            float maxFootProportion = 7;
            float footProportionA = width / height;
            float footProportionB = height / width;
            return between(footProportionA, minFootProportion, maxFootProportion) || between(footProportionB, minFootProportion,
                                                                                     maxFootProportion);
        }
        
        public Image<Bgr, byte> LoadFromFile(string name)
        {
            return new Image<Bgr, byte>(".\\" + name);   
        }

        public bool isFootSize(RotatedRect box)
        {
            float width = box.Size.Width;
            float height = box.Size.Height;
            float footSize = width * height;
            return isFootProprtion(width, height) && between(footSize, 3000, 16000);
        }

        public bool between(float value, float min, float max){
            return (value > min) && (value < max);
        }

        public List<RotatedRect> detect(Image<Bgr, byte> frame)
        {
            var tresholdValue = 255;
            List<RotatedRect> boxes = new List<RotatedRect>();
            Image<Gray, byte> imageGray = frame.Convert<Gray, byte>();
            using (var filtered = filter(imageGray))
            using (var tresholdImage = filtered.ThresholdAdaptive(new Gray(220), AdaptiveThresholdType.GaussianC, ThresholdType.Binary, 55, new Gray(3)))
            {
                
                var conturs = new VectorOfVectorOfPoint();
                CvInvoke.FindContours(tresholdImage, conturs, null, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);
                for (int i = 0; i < conturs.Size; i++)
                {
                    using(VectorOfPoint contour = conturs[i])
                    {
                        if (contour.Size < 100)
                        {
                            continue;
                        }
                        var box = CvInvoke.FitEllipse(contour);
                        if (isCorrectBox(box) && isFootSize(box))
                        {
                            boxes.Add(box);
                        }
                    }
                }
                
            }
            return boxes;
        }

        private bool isCorrectBox(RotatedRect rect)
        {
            return rect.Size.Width > 0 && rect.Size.Height > 0;
        }

        private Image<Gray, byte> filter(Image<Gray, byte> image)
        {
            Image<Gray, byte> filtered = new Image<Gray, byte>(image.Size);
            var k = 0.44f;
            Matrix<float> kernel = new Matrix<float>(new[,] {
                { k, k, k },
                { k, k, k },
                { k, k, k } });
            CvInvoke.Filter2D(image, filtered, kernel, new Point(-1, -1));
            Image<Gray, byte> output = new Image<Gray, byte>(image.Size);

            CvInvoke.MedianBlur(filtered, output, 7);
            return output;
        }
        
    }
}
