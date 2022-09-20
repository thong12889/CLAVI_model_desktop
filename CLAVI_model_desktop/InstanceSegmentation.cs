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
    class InstanceSegmentation : IDisposable
    {
        private InferenceSession sess = null;

        public InstanceSegmentation(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sess = new InferenceSession(modelPath, option);
        }
        public Mat insegInference(Mat image, string labelPath, float threshold, double opacity)
        {
            float nmsThresh = 0.4f;
            int inputW = 1333;
            int inputH = 800;
            Size imgSize = new Size(inputW, inputH);
            Mat result = image.Clone();

            //Label file
            var label = File.ReadLines(labelPath);
            var labelList = label.ToArray();
            var pallete = GenPalette(labelList.Length);

            Mat imageFloat = image.Resize(imgSize);
            imageFloat = DataPreprocessing(imageFloat);
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
                //Mask
                var mask_value = resultsArray[2].AsEnumerable<float>().ToArray();
                var mask_dim = resultsArray[2].AsTensor<float>().Dimensions.ToArray();

                var (candidate, label_out, mask_out) = GetInstanceCandidate(pred_value, pred_dim, label_value, mask_value, mask_dim, threshold);

                if(candidate.Count != 0)
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

                    if(indices != null)
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

                            var mask = mask_out[idx].Resize(new Size(rescale_Xmax - rescale_Xmin, rescale_Ymax - rescale_Ymin));
                            var gray = new Mat();
                            Cv2.CvtColor(mask, gray, ColorConversionCodes.BGR2GRAY);
                            OpenCvSharp.Point[][] contours;
                            OpenCvSharp.HierarchyIndex[] hindex;
                            Cv2.FindContours(gray, out contours, out hindex, RetrievalModes.CComp, ContourApproximationModes.ApproxNone);
                            for(int i = 0; i < contours[0].Length; i++)
                            {
                                contours[0][i].X = contours[0][i].X + (int)rescale_Xmin;
                                contours[0][i].Y = contours[0][i].Y + (int)rescale_Ymin;
                            }
                            Cv2.FillPoly(image, contours, new Scalar(pallete[label_value[idx]].Item0, pallete[label_value[idx]].Item1, pallete[label_value[idx]].Item2));
                            Cv2.DrawContours(image, contours, -1, new Scalar(255, 255, 255), 1);
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
                    Cv2.AddWeighted(image, opacity, result, 1 - opacity, 0, result);
                }
            }
            return result;
        }
        public static (List<List<float>>, List<int>, List<Mat>) GetInstanceCandidate(float[] pred, int[] pred_dim, long[] labels, float[] masks, int[] masks_dim, float pred_thresh = 0.25f)
        {
            List<List<float>> candidate = new List<List<float>>();
            List<int> labelCand = new List<int>();
            List<Mat> masksCand = new List<Mat>();
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
                        //pred
                        List<float> pred_Cand_value = new List<float>();
                        for (int i = 0; i < pred_dim[2]; i++)
                        {
                            int sub_idx = idx1 + i;
                            pred_Cand_value.Add(pred[sub_idx]);
                        }
                        candidate.Add(pred_Cand_value);

                        int idlabelmask = idx1 / 5;
                        //labels
                        labelCand.Add((int)labels[idlabelmask]);

                        //masks
                        int cls_id = (int)labels[idlabelmask];
                        var maskMats = ConvertInstanceSegmentationResult(idlabelmask, masks, masks_dim, pred_thresh);
                        masksCand.Add(maskMats);
                    }
                }
            }
            return (candidate, labelCand, masksCand);
        }
        public static Mat ConvertInstanceSegmentationResult(int id, float[] pred, int[] pred_dim, float threshold)
        {
            
            Mat mat = new Mat(new Size(pred_dim[3], pred_dim[2]), MatType.CV_8UC3);
            for (int batch = 0; batch < pred_dim[1]; batch++)
            {
                for (int h = 0; h < pred_dim[2]; h++)
                {
                    for (int w = 0; w < pred_dim[3]; w++)
                    {
                        int idx = (batch * pred_dim[1] * pred_dim[2] * pred_dim[3]) + (id * pred_dim[2] * pred_dim[3]) + (h * pred_dim[3]) + w;

                        Vec3b pix = mat.At<Vec3b>(h, w);
                        if (pred[idx] < threshold)
                        {
                            pix = new Vec3b(0, 0, 0);
                        }
                        else
                        {
                            pix = new Vec3b(255, 255, 255);
                        }
                        mat.Set<Vec3b>(h, w, pix);
                    }
                }
            }
            return mat;
        }
        private Mat DataPreprocessing(Mat image)
        {
            Mat data = Mat.Zeros(image.Size(), MatType.CV_32FC3);
            using (var rgbImage = new Mat())
            {
                Cv2.CvtColor(image, rgbImage, ColorConversionCodes.BGR2RGB);
                rgbImage.ConvertTo(data, MatType.CV_32FC3, (float)(1 / 255.0));
                var channelData = Cv2.Split(data);
                channelData[0] = (channelData[0] - 0.485) / 0.229;
                channelData[1] = (channelData[1] - 0.456) / 0.224;
                channelData[2] = (channelData[2] - 0.406) / 0.225;
                Cv2.Merge(channelData, data);
            }
            return data;
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
