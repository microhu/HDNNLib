using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.WindowsAzure.IntelligentServices.Pronunciation.AlgorithmHelper;
using System.IO;

namespace HDNNLib_Test
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 4)
            {
                Console.WriteLine("exe modelFolder idWavPathScp idScriptScp outDir");
                return;
            }

            string modelFolder = args[0];
            string idWavPathScp = args[1];
            string idScriptScp = args[2];
            string outDir = args[3];

            if (!Directory.Exists(outDir)) Directory.CreateDirectory(outDir);

            Dictionary<string, string> idWavPair = readKeyValuePair(idWavPathScp);
            Dictionary<string, string> idScriptPair = readKeyValuePair(idScriptScp);

            ScoringHost.Initialize(modelFolder);

            foreach (var pair in idWavPair)
            {
                if (idScriptPair.Keys.Contains(pair.Key))
                {
                    string outFile = Path.Combine(outDir, pair.Key + ".log");
                   

                    byte [] wavData=File.ReadAllBytes(pair.Value);
                    try
                    {

                        SentenceEvaluationResult scoring = ScoringHost.Evaluate(
                        SupportedLanguage.English,
                        idScriptPair[pair.Key],
                        wavData
                        );

                        List<string> outLines = new List<string>();
                        foreach (var w in scoring.Words)
                        {
                            string wres = string.Format("{0} {1} {2} {3}", w.StartTime, w.EndTime,  w.Word.Trim(new char[]{' ','\t','\r','\n'}), w.Score);
                            outLines.Add(wres);
                        }
                        writeVecContentIntoFile(outLines, outFile);
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Incorrect for our DLL!" + e.Message);
                    }
                }
                else
                {
                    Console.WriteLine("no script for sentence {0}", pair.Key);
                }
            }


        }

        private static void writeVecContentIntoFile(List<string> outLines, string outFile)
        {
            StreamWriter sw = File.CreateText(outFile);
            foreach (string line in outLines) sw.WriteLine(line);
            sw.Close();
        }

        private static Dictionary<string, string> readKeyValuePair(string idWavPathScp)
        {
            Dictionary<string, string> keyValuePairs = new Dictionary<string, string>();
            StreamReader sr = File.OpenText(idWavPathScp);
            string[] lines = sr.ReadToEnd().Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            sr.Close();
            foreach (string line in lines)
            {
                string[] toks = line.Split(new char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (toks.Length == 2)
                {
                    keyValuePairs.Add(toks[0], toks[1]);
                }
                else
                {
                    Console.WriteLine("Error format for {0}", line);
                }
            }
            return keyValuePairs;
        }
    }
}
