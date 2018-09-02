namespace Microsoft.WindowsAzure.IntelligentServices.Pronunciation.AlgorithmHelper
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Runtime.InteropServices;

    public static class FeatureGenerator
    {
        private static IEnumerable<float> PreprocessF0(double[] f0Data, bool differentrate)
        {
            if (differentrate)
            {
                for (var i = 0; i + 1 < f0Data.Length; i += 2)
                {
                    yield return (float)(0.5 * (f0Data[i] + f0Data[i + 1]));
                }
            }
            else
            {
                for (var i = 0; i < f0Data.Length; i++)
                {
                    yield return (float)f0Data[i];
                }
            }
        }

        private static IEnumerable<float> DeltaValues(float[] chunk)
        {
            Func<int, float> map = index =>
                {
                    if (index < 0)
                    {
                        return chunk[0];
                    }
                    if (index >= chunk.Length)
                    {
                        return chunk[chunk.Length - 1];
                    }
                    return chunk[index];
                };

            for (var i = 0; i < chunk.Length; i++)
            {
                yield return (2.0f * (map(i + 2) - map(i - 2)) + (map(i + 1) - map(i - 1))) * 0.1f;
            }
        }

        private static IList<float>[] PitchAndDeltasCalcuate(float[] pitches)
        {
            List<float>[] pitchDeltas = { new List<float>(), new List<float>(), new List<float>() };

            IList<Tuple<int, int>> ranges = new List<Tuple<int, int>>();
            {
                var startIndex = 0;
                var endIndex = 0;
                var zeroFlag = true;

                while (endIndex != pitches.Length)
                {
                    if ((pitches[endIndex] > 1.0) == zeroFlag)
                    {
                        ranges.Add(new Tuple<int, int>(startIndex, endIndex - startIndex));
                        startIndex = endIndex;
                        zeroFlag = !zeroFlag;
                    }
                    endIndex++;
                }

                ranges.Add(new Tuple<int, int>(startIndex, endIndex - startIndex));
            }

            foreach (var range in ranges)
            {
                var pitchChunk = pitches.Skip(range.Item1).Take(range.Item2).ToArray();

                if (pitchChunk.Length < 3 || pitchChunk[0] < 1.0f)
                {
                    var zeroChunk = Enumerable.Repeat(0.0f, pitchChunk.Length).ToArray();
                    pitchDeltas[0].AddRange(zeroChunk);
                    pitchDeltas[1].AddRange(zeroChunk);
                    pitchDeltas[2].AddRange(zeroChunk);
                }
                else
                {
                    var chunk = pitchChunk.Select(p => (float)Math.Log10(p)).ToArray();
                    pitchDeltas[0].AddRange(chunk);
                    chunk = DeltaValues(chunk).ToArray();
                    pitchDeltas[1].AddRange(chunk);
                    chunk = DeltaValues(chunk).ToArray();
                    pitchDeltas[2].AddRange(chunk);
                }
            }

            return pitchDeltas;
        }

        private static void ReadMfcFeature(byte[] mfcData, out Header header, out IList<float> frames)
        {
            header = new Header();
            frames = new List<float>();

            using (var reader = new BinaryReader(new MemoryStream(mfcData)))
            {
                header.NatureOrder = true;
                header.Samples = reader.ReadInt32();
                header.SamplePeriod = reader.ReadInt32();
                header.SampleSizeInBytes = reader.ReadInt16();
                header.ParmKind = reader.ReadInt16();

                var compressed = (header.ParmKind & 1024) != 0;
                var byteValue = compressed ? sizeof(short) : sizeof(float);
                header.SampleSize = (short)(header.SampleSizeInBytes / byteValue);

                IList<float> a = Enumerable.Repeat(1.0f, header.SampleSize).ToList();
                IList<float> b = Enumerable.Repeat(0.0f, header.SampleSize).ToList();

                if (compressed)
                {
                    header.Samples -= 4;
                    for (var i = 0; i < header.SampleSize; i++)
                    {
                        a[i] = reader.ReadSingle();
                    }
                    for (var i = 0; i < header.SampleSize; i++)
                    {
                        b[i] = reader.ReadSingle();
                    }
                }

                for (var i = 0; i < header.Samples; i++)
                {
                    for (var j = 0; j < header.SampleSize; j++)
                    {
                        if (compressed)
                        {
                            frames.Add((reader.ReadInt16() + b[j]) / a[j]);
                        }
                        else
                        {
                            frames.Add(reader.ReadSingle());
                        }
                    }
                }
            }
        }

        private static IEnumerable<float> MergeDataString(
            IList<float>[] pitchDeltas,
            int ndim,
            IList<float> frames,
            int mfcdim)
        {
            var pIndex = 0;
            for (var i = 0; i < frames.Count; i++)
            {
                if (i % ndim < mfcdim)
                {
                    yield return frames[i];
                }
                if (i % ndim == mfcdim - 1)
                {
                    if (pIndex < pitchDeltas[0].Count)
                    {
                        yield return pitchDeltas[0][pIndex];
                        yield return pitchDeltas[1][pIndex];
                        yield return pitchDeltas[2][pIndex];
                    }
                    else
                    {
                        yield return 0.0f;
                        yield return 0.0f;
                        yield return 0.0f;
                    }
                    pIndex++;
                }
            }
        }

        private static byte[] ToHtkString(Header header, IEnumerable<float> effects)
        {
            using (var writer = new BinaryWriter(new MemoryStream()))
            {
                writer.Write(header.Samples);
                writer.Write(header.SamplePeriod);
                writer.Write(header.SampleSizeInBytes);
                writer.Write(header.ParmKind);

                foreach (var f in effects)
                {
                    writer.Write(f);
                }

                return (writer.BaseStream as MemoryStream).ToArray();
            }
        }

        public static double[] ExtractF0Feature(byte[] wavData)
        {
            IntPtr ptr;
            var length = NativeMethods.ExtractF0Feature(wavData, wavData.Length, out ptr);
            var data = new double[length / sizeof(double)];
            Marshal.Copy(ptr, data, 0, length / sizeof(double));

            return data;
        }

        public static byte[] AppendF0Feature(double[] f0Data, byte[] mfcData)
        {
            var effectiveF0Data = PreprocessF0(f0Data, false).ToArray();
            var pitchDeltas = PitchAndDeltasCalcuate(effectiveF0Data);

            Header header;
            IList<float> frames;
            ReadMfcFeature(mfcData, out header, out frames);
            var effects = MergeDataString(pitchDeltas, header.SampleSize, frames, header.SampleSize);

            header.SampleSize += (short)pitchDeltas.Length;
            header.ParmKind = 9;

            var compressed = (header.ParmKind & 1024) != 0;
            var byteValue = compressed ? sizeof(short) : sizeof(float);
            header.SampleSizeInBytes = (short)(header.SampleSize * byteValue);

            mfcData = ToHtkString(header, effects);

            return mfcData;
        }

        public static byte[] ExtractMfcFeature(byte[] wavData)
        {
            byte[] mfcData;
            IntPtr mfcDataPtr;
            var length = NativeMethods.ExtractMfcFeature(wavData, wavData.Length, out mfcDataPtr);
            mfcData = new byte[length];
            Marshal.Copy(mfcDataPtr, mfcData, 0, mfcData.Length);

            return mfcData;
        }

        private class Header
        {
            public int Samples { get; set; }

            public int SamplePeriod { get; set; }

            public short SampleSizeInBytes { get; set; }

            public short SampleSize { get; set; }

            public short ParmKind { get; set; }

            public bool NatureOrder { get; set; }
        }
    }
}