/********************************************************
*                                                        *
*   Copyright (c) Microsoft. All rights reserved.        *
*                                                        *
*********************************************************/

namespace Microsoft.WindowsAzure.IntelligentServices.Pronunciation.AlgorithmHelper
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.Serialization;

    [Serializable]
    [DataContract]
    public class SentenceEvaluationResult
    {
        [DataMember(Name = "score")]
        public double Score { get; set; }

        [DataMember(Name = "words")]
        public List<WordEvaluationResult> Words { get; set; }
    }
}