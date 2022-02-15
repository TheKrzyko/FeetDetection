using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace FeetDetectionLibrary
{
    public class FeetDetection
    {
        private readonly float MIN_FOOT_PROPORTION = 2.1f;
        private readonly float MAX_FOOT_PROPORTION = 5.0f;
        private readonly float MIN_FOOT_AREA = 0.009f;
        private readonly float MAX_FOOT_AREA = 0.02f;

        public class Preprocess
        {
            private bool doPerspective = false;
            private Mat warpMatrix = null;
            public void Callibrate(string inputPerspectiveFilename) {

                int[] srcv = File.ReadAllText(inputPerspectiveFilename).Split('\n', ',').Select(str => int.Parse(str)).ToArray();
                PointF[] src = new PointF[4];
                src[0] = new PointF(srcv[0], srcv[1]);
                src[1] = new PointF(srcv[2], srcv[3]);
                src[2] = new PointF(srcv[4], srcv[5]);
                src[3] = new PointF(srcv[6], srcv[7]);
                int[] dstv = { 0, 0, 2400, 0, 0, 2400, 2400, 2400 };
                PointF[] dst = new PointF[4];
                dst[0] = new PointF(dstv[0], dstv[1]);
                dst[1] = new PointF(dstv[2], dstv[3]);
                dst[2] = new PointF(dstv[4], dstv[5]);
                dst[3] = new PointF(dstv[6], dstv[7]);
                this.CallibratePerspective(src,dst);
            }

            public void TestCallibrate()
            {
                int[] srcv = { 1300, 574,2834, 302,1114, 2521,2596, 3136};
                PointF[] src = new PointF[4];
                src[0] = new PointF(srcv[0], srcv[1]);
                src[1] = new PointF(srcv[2], srcv[3]);
                src[2] = new PointF(srcv[4], srcv[5]);
                src[3] = new PointF(srcv[6], srcv[7]);

                int[] dstv =  {0,0,2400,0,0,2400,2400,2400};
                PointF[] dst = new PointF[4];
                dst[0] = new PointF(dstv[0], dstv[1]);
                dst[1] = new PointF(dstv[2], dstv[3]);
                dst[2] = new PointF(dstv[4], dstv[5]);
                dst[3] = new PointF(dstv[6], dstv[7]);
                this.CallibratePerspective(src,dst);

            }
            public void CallibratePerspective(PointF[] src, PointF[] dst)
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

        private bool isFootProportion(float width, float height)
        {
            var min = Math.Min(width, height);
            var max = Math.Max(width, height);
            return between(max/min, MIN_FOOT_PROPORTION, MAX_FOOT_PROPORTION);
        }

        private bool isFootSize(RotatedRect box, float frameArea)
        {
            float width = box.Size.Width;
            float height = box.Size.Height;
            float footArea = width * height;
            return isFootProportion(width, height) && between(footArea/frameArea, MIN_FOOT_AREA, MAX_FOOT_AREA);
        }

        private bool between(float value, float min, float max){
            return (value > min) && (value < max);
        }

        private RotatedRect scaleRelativeToSize(RotatedRect rect, SizeF size)
        {
            rect.Size.Width /= size.Width;
            rect.Size.Height /= size.Height;
            rect.Center.X /= size.Width;
            rect.Center.Y /= size.Height;
            return rect;
        }

        /// <summary>
        /// Detects feet ellipses in image.
        /// </summary>
        /// <param name="frame">Preprocessed input image</param>
        /// <returns>List of RotatedRect, represented as coordinates in range 0 to 1.</returns>
        public List<RotatedRect> detect(Image<Bgr, byte> frame)
        {
            var rects = detectAsPixelCoord(frame);
            for (int i = 0; i < rects.Count; i++)
            {
                rects[i] = scaleRelativeToSize(rects[i], frame.Size);
            }
            return rects;
        }

        /// <summary>
        /// Detects feet ellipses in image.
        /// </summary>
        /// <param name="frame">Preprocessed input image</param>
        /// <returns>List of RotatedRect, represented as pixel coordinates of input frame</returns>
        public List<RotatedRect> detectAsPixelCoord(Image<Bgr, byte> frame)
        {
            Image<Gray, byte> t;
            return detectAsPixelCoord(frame, out t);
        }

        public List<RotatedRect> detectAsPixelCoord(Image<Bgr, byte> frame, out Image<Gray, byte> tresholdImage)
        {
            List<RotatedRect> boxes = new List<RotatedRect>();
            Image<Gray, byte> imageGray = frame.Convert<Gray, byte>();
            var frameArea = frame.Width * frame.Height;
            using (var filtered = filter(imageGray))
            {
                tresholdImage = filtered.ThresholdAdaptive(new Gray(255), AdaptiveThresholdType.GaussianC, ThresholdType.Binary, 55, new Gray(5));
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
                        if (isCorrectBox(box) && isFootSize(box, frameArea))
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
