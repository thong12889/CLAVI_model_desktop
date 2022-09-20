using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace CLAVI_model_desktop
{
    class ObjectDetection : IDisposable
    {
        private InferenceSession sess = null;

        public ObjectDetection(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sess = new InferenceSession(modelPath, option);
        }
        public Mat ObjInference(Mat image, string labelPath, float threshold)
        {
            float nmsThresh = 0.4f;
            int inputW = 640;
            int inputH = 640;
            Size imgSize = new Size(inputW, inputH);

            //Label file
            var label = File.ReadLines(labelPath);
            var labelList = label.ToArray();
            var pallete = GenPalette(labelList.Length);

            Mat imageFloat = image.Resize(imgSize);
            imageFloat.ConvertTo(imageFloat, MatType.CV_32FC1);
            var input = new DenseTensor<float>(MatToList(imageFloat), new[] { 1, 3, imgSize.Height, imgSize.Width });

            // Setup inputs and outputs
            var inputMeta = sess.InputMetadata;
            var inputName = inputMeta.Keys.ToArray()[0];
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, input)
            };

            using (var results = sess.Run(inputs))
            {
                //Postprocessing
                var resultsArray = results.ToArray();
                //Pred
                var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
                var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
                //Label
                var label_value = resultsArray[1].AsEnumerable<Int64>().ToArray();

                //Fillter by score
                var candidate = GetCandidate(pred_value, pred_dim, threshold);

                if (candidate.Count != 0)
                {
                    //NMS
                    List<Rect> bboxes = new List<Rect>();
                    List<float> confidences = new List<float>();
                    for (int i = 0; i < candidate.Count; i++)
                    {
                        Rect box = new Rect((int)candidate[i][0], (int)candidate[i][1],
                           (int)(candidate[i][2] - candidate[i][0]), (int)(candidate[i][3] - candidate[i][1]));
                        bboxes.Add(box);
                        confidences.Add(candidate[i][4]);
                    }
                    int[] indices;
                    CvDnn.NMSBoxes(bboxes, confidences, threshold, nmsThresh, out indices);

                    if (indices != null)
                    {
                        for (int ids = 0; ids < indices.Length; ids++)
                        {
                            int idx = indices[ids];

                            var dw = image.Width / (float)imgSize.Width;
                            var dh = image.Height / (float)imgSize.Height;

                            var rescale_Xmin = candidate[idx][0] * dw;
                            var rescale_Ymin = candidate[idx][1] * dh;
                            var rescale_Xmax = candidate[idx][2] * dw;
                            var rescale_Ymax = candidate[idx][3] * dh;

                            //draw bounding box
                            Cv2.Rectangle(image, new Rect((int)rescale_Xmin, (int)rescale_Ymin, (int)(rescale_Xmax - rescale_Xmin), (int)(rescale_Ymax - rescale_Ymin)),
                                new Scalar(pallete[label_value[idx]].Item0, pallete[label_value[idx]].Item1, pallete[label_value[idx]].Item2), 5);
                        }
                        for (int ids = 0; ids < indices.Length; ids++)
                        {
                            int idx = indices[ids];
                            var confi = candidate[idx][4];

                            var dw = image.Width / (float)imgSize.Width;
                            var dh = image.Height / (float)imgSize.Height;

                            var rescale_Xmin = candidate[idx][0] * dw;
                            var rescale_Ymin = candidate[idx][1] * dh;

                            //draw label
                            var result_text = labelList[label_value[idx]] + "|" + confi.ToString("0.00");
                            var scale = 0.8;
                            var thickness = 1;
                            HersheyFonts font = HersheyFonts.HersheyDuplex;
                            int baseLine;
                            var textSize = Cv2.GetTextSize(result_text, font, scale, thickness, out baseLine);
                            Cv2.Rectangle(image, new Point(rescale_Xmin + 4, rescale_Ymin + 4), new Point(rescale_Xmin + textSize.Width, rescale_Ymin + textSize.Height + 10),
                                new Scalar(0, 0, 0), -1);
                            Cv2.PutText(image, result_text, new Point(rescale_Xmin + 4, rescale_Ymin + textSize.Height + 4), font, scale,
                                new Scalar(255, 255, 255), thickness);
                        }
                    }
                }
            }
            return image;
        }
        public static List<List<float>> GetCandidate(float[] pred, int[] pred_dim, float pred_thresh = 0.25f)
        {
            List<List<float>> candidate = new List<List<float>>();
            for (int batch = 0; batch < pred_dim[0]; batch++)
            {
                for (int cand = 0; cand < pred_dim[1]; cand++)
                {
                    int score = 4;//Default 4  // object ness score
                    int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
                    int idx2 = idx1 + score;
                    var value = pred[idx2];
                    if (value > pred_thresh)
                    {
                        List<float> tmp_value = new List<float>();
                        for (int i = 0; i < pred_dim[2]; i++)
                        {
                            int sub_idx = idx1 + i;
                            tmp_value.Add(pred[sub_idx]);
                        }
                        candidate.Add(tmp_value);
                    }
                }
            }
            return candidate;
        }
        static Vec3b[] GenPalette(int classes)
        {
            Random rnd = new Random(classes);
            Vec3b[] palette = new Vec3b[classes];
            for (int i = 0; i < classes; i++)
            {
                byte v1 = (byte)rnd.Next(0, 255);
                byte v2 = (byte)rnd.Next(0, 255);
                byte v3 = (byte)rnd.Next(0, 255);
                palette[i] = new Vec3b(v1, v2, v3);
            }
            return palette;
        }
        private static float[] MatToList(Mat mat)
        {
            var ih = mat.Height;
            var iw = mat.Width;
            var chn = mat.Channels();
            unsafe
            {
                return Create((float*)mat.DataPointer, ih, iw, chn);
            }
        }
        private unsafe static float[] Create(float* ptr, int ih, int iw, int chn)
        {
            float[] array = new float[chn * ih * iw];

            for (int y = 0; y < ih; y++)
            {
                for (int x = 0; x < iw; x++)
                {
                    for (int c = 0; c < chn; c++)
                    {
                        var idx = (y * chn) * iw + (x * chn) + c;
                        var idx2 = (c * iw) * ih + (y * iw) + x;
                        array[idx2] = ptr[idx];
                    }
                }
            }
            return array;
        }
        public void Dispose()
        {
            sess?.Dispose();
            sess = null;
        }
    }
}
