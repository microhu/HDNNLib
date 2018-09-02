namespace Microsoft.WindowsAzure.IntelligentServices.Pronunciation.AlgorithmHelper
{
    using System;
    using System.Runtime.InteropServices;
    using System.Text;

    public static class NativeMethods
    {
        // FeatureExtraction

        [DllImport("FeatureExtraction.dll", EntryPoint = "Initialization", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int InitializeMfcExtraction(string command);

        [DllImport("FeatureExtraction.dll", EntryPoint = "UnInitializationFeatureExaction", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int UnInitializationFeatureExaction();

        [DllImport("FeatureExtraction.dll", EntryPoint = "FeatureExtractionFromMemory", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ExtractMfcFeature(byte[] intputData, int length, out IntPtr mfcPointer);

        // F0Extraction

        [DllImport("F0Extraction.dll", EntryPoint = "F0ExtractionInitialization", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int InitializeF0Extraction(string command);

        [DllImport("F0Extraction.dll", EntryPoint = "F0ExtractionFromMemory", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ExtractF0Feature(byte[] inputData, int length, out IntPtr f0Pointer);

        [DllImport("F0Extraction.dll", EntryPoint = "UnF0ExtractionInitialization", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int UnF0ExtractionInitialization();

        // DNN_HMM

        [DllImport("DNN_HMM.dll", EntryPoint = "LoadModel", CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern int LoadModel(string modelFolder);

        [DllImport("DNN_HMM.dll", EntryPoint = "unLoadModel", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int unLoadModel();

        [DllImport("DNN_HMM.dll", EntryPoint = "EvaluateStrictBoundary", CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Evaluate(
            byte[] mfcData,
            [Out] [MarshalAs(UnmanagedType.LPWStr)] StringBuilder resultBuffer,
            int resultBufferLength,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPWStr)] string[] words,
            int wordsLength);

        [DllImport("DNN_HMM.dll", EntryPoint = "EvaluateWithCompetingPhonesMemoStream", CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Evaluate_competingPhones(
            byte[] mfcData,
            [Out] [MarshalAs(UnmanagedType.LPWStr)] StringBuilder resultBuffer,
            int resultBufferLength,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPWStr)] string[] words,
            int wordsLength,
            int competingPhoneNumber,
            bool fixPhoneBoundary);

    }
}