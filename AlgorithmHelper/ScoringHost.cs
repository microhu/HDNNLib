namespace Microsoft.WindowsAzure.IntelligentServices.Pronunciation.AlgorithmHelper
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// The ScoringHost. Leveraged from the workrole of CaptainService.
    /// </summary>
    public static class ScoringHost
    {
        private static ICollection<string> phonemeDictionary;

        private const int EvaluateResultMinLength = 1024 * 2;

        private const int PhonemeResultMaxLength = 50;

        public static void Initialize(string modelFolder)
        {
            Trace.TraceInformation("ScoringHost::Initialize(), Path=" + modelFolder);
            try
            {
                NativeMethods.LoadModel(modelFolder); // alignment by likelihood
                NativeMethods.InitializeMfcExtraction(string.Format("HTKfunctions -C {0}", Path.Combine(modelFolder, "hcopy.config")));
                NativeMethods.InitializeF0Extraction(
                    string.Format("F0Extraction -C {0}", Path.Combine(modelFolder, "getf0.config")));
                phonemeDictionary = new HashSet<string>(ParsePhonemeDictionay(Path.Combine(modelFolder, "evadict")));
            }
            catch (Exception e)
            {
                Trace.TraceError(
                    "ScoringHost::Initialize(), unexpected exception, Message={0}, Stack={1}",
                    e.Message,
                    e.StackTrace);

                throw;
            }
        }

        public static void UnInitialize()
        {
            //NativeMethods.UnF0ExtractionInitialization();
            //NativeMethods.UnInitializationFeatureExaction();
            NativeMethods.unLoadModel();
        }

        public static SentenceEvaluationResult Evaluate_competingPhones(SupportedLanguage language, string script, byte[] wavData)
        {
            Trace.TraceInformation(DateTime.Now.ToString("yyyyMMdd-HHmmss.fff") + " evaluate started " + System.Threading.Thread.CurrentThread.ManagedThreadId);
            var wordList = script.Split(new[] { " " }, StringSplitOptions.RemoveEmptyEntries);
            var words = wordList.Select(w =>w).ToArray();
            

            var mfcData = FeatureGenerator.ExtractMfcFeature(wavData);
            if (mfcData == null || mfcData.Length == 0)
            {
                Trace.TraceWarning("ScoringHost::Evaluate(), mfc extraction error!");
                throw new InvalidDataException("mfc extraction error!");
            }

            if (SupportedLanguage.Chinese == language)
            {
                double[] f0Data = FeatureGenerator.ExtractF0Feature(wavData);
                if (f0Data == null || f0Data.Length == 0)
                {
                    Trace.TraceWarning("ScoringHost::Evaluate(), f0 extraction error!");
                    throw new InvalidDataException("f0 extraction error!");
                } 
                
                mfcData = FeatureGenerator.AppendF0Feature(f0Data, mfcData);
            }

            IList<string> wordsWithSilence = new List<string>(words);
            wordsWithSilence.Insert(0, "<s>");
            wordsWithSilence.Add("</s>");

            int maxLength = 0;
            if (language == SupportedLanguage.English)
            {
                maxLength = PhonemeResultMaxLength * 12 * words.Length;
            }
            else if (language == SupportedLanguage.Chinese)
            {
                maxLength = PhonemeResultMaxLength * 5 * words.Length;
            }

            if (maxLength < EvaluateResultMinLength)
            {
                maxLength = EvaluateResultMinLength;
            }

            var resultBuffer = new StringBuilder(maxLength);
            var resultLength = NativeMethods.Evaluate_competingPhones(
                mfcData,
                resultBuffer,
                maxLength,
                wordsWithSilence.ToArray(),
                wordsWithSilence.Count,
                5,
                true);
            /*
            var resultLength = NativeMethods.Evaluate(
                mfcData,
                resultBuffer,
                maxLength,
                wordsWithSilence.ToArray(),
                wordsWithSilence.Count);            
            */
            try
            {
                if (resultLength < 0)
                {
                    if (resultLength == -1)
                    {
                        throw new Exception("Script and Wave not match!");
                    }
                }

                var phonemeEvaluationResults = resultBuffer.ToString(0, resultLength).Split(new[] { "\n" }, StringSplitOptions.RemoveEmptyEntries);

                Trace.TraceInformation(DateTime.Now.ToString("yyyyMMdd-HHmmss.fff") + " evaluate ended " + System.Threading.Thread.CurrentThread.ManagedThreadId);

                return CreateSentenceEvaluationResult(phonemeEvaluationResults);
            }
            catch (Exception e)
            {
                Trace.TraceError(
                    "ScoringHost::Evaluate(), unexpected exception while parsing, Message={0}, StackTrace={1}",
                    e.Message,
                    e.StackTrace);
                throw;
            }
        }

        public static SentenceEvaluationResult Evaluate(SupportedLanguage language, string script, byte[] wavData)
        {
            Trace.TraceInformation(DateTime.Now.ToString("yyyyMMdd-HHmmss.fff") + " evaluate started " + System.Threading.Thread.CurrentThread.ManagedThreadId);
            var wordList = script.Split(new[] { " " }, StringSplitOptions.RemoveEmptyEntries);
            var words = wordList.Select(w => language == SupportedLanguage.English ? w.ToUpper() : w).ToArray();
            foreach (var word in words)
            {
                if (!phonemeDictionary.Contains(word))
                {
                    Trace.TraceWarning("ScoringHost::Evaluate(), unable to identify the word, Content={0}", word);
                    throw new ArgumentOutOfRangeException(word);
                }
            }

            var mfcData = FeatureGenerator.ExtractMfcFeature(wavData);
            if (mfcData == null || mfcData.Length == 0)
            {
                Trace.TraceWarning("ScoringHost::Evaluate(), mfc extraction error!");
                throw new InvalidDataException("mfc extraction error!");
            }

            if (SupportedLanguage.Chinese == language)
            {
                double[] f0Data = FeatureGenerator.ExtractF0Feature(wavData);
                if (f0Data == null || f0Data.Length == 0)
                {
                    Trace.TraceWarning("ScoringHost::Evaluate(), f0 extraction error!");
                    throw new InvalidDataException("f0 extraction error!");
                }

                mfcData = FeatureGenerator.AppendF0Feature(f0Data, mfcData);
            }

            IList<string> wordsWithSilence = new List<string>(words);
            wordsWithSilence.Insert(0, "<s>");
            wordsWithSilence.Add("</s>");

            int maxLength = 0;
            if (language == SupportedLanguage.English)
            {
                maxLength = PhonemeResultMaxLength * 12 * words.Length;
            }
            else if (language == SupportedLanguage.Chinese)
            {
                maxLength = PhonemeResultMaxLength * 5 * words.Length;
            }

            if (maxLength < EvaluateResultMinLength)
            {
                maxLength = EvaluateResultMinLength;
            }

            var resultBuffer = new StringBuilder(maxLength);
            var resultLength = NativeMethods.Evaluate(
                mfcData,
                resultBuffer,
                maxLength,
                wordsWithSilence.ToArray(),
                wordsWithSilence.Count);            
            
            try
            {
                if (resultLength < 0)
                {
                    if (resultLength == -1)
                    {
                        throw new Exception("Script and Wave not match!");
                    }
                }

                var phonemeEvaluationResults = resultBuffer.ToString(0, resultLength).Split(new[] { "\n" }, StringSplitOptions.RemoveEmptyEntries);

                Trace.TraceInformation(DateTime.Now.ToString("yyyyMMdd-HHmmss.fff") + " evaluate ended " + System.Threading.Thread.CurrentThread.ManagedThreadId);

                return CreateSentenceEvaluationResult(phonemeEvaluationResults);
            }
            catch (Exception e)
            {
                Trace.TraceError(
                    "ScoringHost::Evaluate(), unexpected exception while parsing, Message={0}, StackTrace={1}",
                    e.Message,
                    e.StackTrace);
                throw;
            }
        }

        public static double GetScorePercentage(double score)
        {
            
            if (score < 0.05)
            {
                return (score / 0.05 * 40);
            }
            else if (score < 0.12)
            {
                return 40 + ((score - 0.05) / 0.07 * 30);
            }
            else if (score < 0.2)
            {
                return 70 + ((score - 0.12) / 0.08 * 20);
            }
            else if (score < 0.36)
            {
                return 90 + ((score - 0.2) / 0.16 * 10);
            }
            else
            {
                return 100.0;
            }
        }

        private static IEnumerable<string> ParsePhonemeDictionay(string dictionaryPath)
        {
            var lines = File.ReadLines(dictionaryPath);
            var enumerator = lines.GetEnumerator();

            enumerator.MoveNext();
            enumerator.MoveNext();

            while (enumerator.MoveNext())
            {
                var line = enumerator.Current;
                var pos = line.IndexOfAny(new[] { ' ', '\t' });
                if (pos != -1)
                {
                    yield return line.Substring(0, pos);
                }
            }
        }

        private static SentenceEvaluationResult CreateSentenceEvaluationResult(string[] phonemeEvaluationResults)
        {
            var validPhonemeEvaluationResults = phonemeEvaluationResults.Select(l => new PhonemeEvaluationResult(l)).Where(p => p.IsValid).GetEnumerator();
            var wordPhonemeEvaluationResults = new List<PhonemeEvaluationResult>();
            var sentenceEvaluationResult = new SentenceEvaluationResult
            {
                Words = new List<WordEvaluationResult>(),
            };

            while (true)
            {
                var eof = !validPhonemeEvaluationResults.MoveNext();

                if ((eof || validPhonemeEvaluationResults.Current.IsStartOfWord) && wordPhonemeEvaluationResults.Count != 0)
                {
                    var phonemesAvgScore = wordPhonemeEvaluationResults.Average(n => n.Score);
                    sentenceEvaluationResult.Words.Add(
                        new WordEvaluationResult
                        {
                            Score = Math.Round(ScoringHost.GetScorePercentage(phonemesAvgScore)),
                            //Score=phonemesAvgScore,
                            StartTime = wordPhonemeEvaluationResults.First().StartTime,
                            EndTime = wordPhonemeEvaluationResults.Last().EndTime,
                            Word = wordPhonemeEvaluationResults.First().Word,
                        });

                    wordPhonemeEvaluationResults.Clear();
                }

                if (eof)
                {
                    break;
                }

                wordPhonemeEvaluationResults.Add(validPhonemeEvaluationResults.Current);
            }

            var wordsAvgScore = sentenceEvaluationResult.Words.Average(w => w.Score);
            sentenceEvaluationResult.Score = Math.Round(wordsAvgScore);
           // sentenceEvaluationResult.Score = Math.Round( GetScorePercentage(wordsAvgScore));
            return sentenceEvaluationResult;
        }
    }
}