namespace Microsoft.WindowsAzure.IntelligentServices.Pronunciation.AlgorithmHelper
{
    using System;
    using System.Linq;

    public class PhonemeEvaluationResult
    {
        private static readonly string[] SlienceFlags = { "sil", "sil[2]", "sil[4]", "silst" };

        private readonly string[] columns;

        public PhonemeEvaluationResult(string line)
        {
            this.columns = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        }

        public bool IsValid
        {
            get
            {
                if (this.columns.Length < 4)
                {
                    return false;
                }

                return SlienceFlags.All(flag => String.CompareOrdinal(this.Phoneme, flag) != 0);
            }
        }

        public float Score
        {
            get
            {
                return float.Parse(this.columns[3]);
            }
        }

        public int StartTime
        {
            get
            {
                return int.Parse(this.columns[0]) / 100000;
            }
        }

        public int EndTime
        {
            get
            {
                return int.Parse(this.columns[1]) / 100000;
            }
        }

        public string Phoneme
        {
            get
            {
                return this.columns[2];
            }
        }

        public bool IsStartOfWord
        {
            get
            {
                if (!this.IsValid)
                {
                    return false;
                }

                if (this.columns.Length == 5)
                {
                    return true;
                }

                return false;
            }
        }

        public string Word
        {
            get
            {
                if (this.IsStartOfWord)
                {
                    return this.columns[4];
                }
                else
                {
                    throw new Exception("Can not get the word because current phoneme is not the start of it.");
                }
            }
        }
    }
}