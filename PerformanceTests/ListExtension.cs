using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceTests
{
    
    public static class ListExtension
    {
        private static Random rnd = new Random();
        public static T PickRandom<T>(this List<T> source)
        {
            return source[rnd.Next(source.Count)];
        }
    }
}
