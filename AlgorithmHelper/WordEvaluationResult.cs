/********************************************************
*                                                        *
*   Copyright (c) Microsoft. All rights reserved.        *
*                                                        *
*********************************************************/

namespace Microsoft.WindowsAzure.IntelligentServices.Pronunciation.AlgorithmHelper
{
    using System;
    using System.Runtime.Serialization;

    [Serializable]
    [DataContract]
    public class WordEvaluationResult
    {
        [DataMember(Name = "word")]
        public string Word { get; set; }

        [DataMember(Name = "score")]
        public double Score { get; set; }

        [DataMember(Name = "startTime")]
        public int StartTime { get; set; }

        [DataMember(Name = "endTime")]
        public int EndTime { get; set; }
    }
}