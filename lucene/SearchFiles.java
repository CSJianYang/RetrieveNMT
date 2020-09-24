import java.io.IOException;
import java.nio.file.Paths;
import java.io.FileWriter;
import java.io.File;
import java.nio.file.Files;
import java.util.Date;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;



import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
 
public class SearchFiles 
{
 
    public static void main(String[] args) throws Exception 
    {   
        String usage = "SearchFiles"
                     + " [-index INDEX_PATH] [-test] [-search] (source or target) [-topN]\n"
                     + "This indexes the documents in DOCS_PATH, creating a Lucene index"
                     + "in INDEX_PATH that can be searched with SearchFiles";
        String indexPath = "index";
        String testPath=null;
        String search=null;
        int topN=10;
        boolean merge = false;
        for(int i=0;i<args.length;i++) {
            if ("-testPath".equals(args[i])) {
                testPath = args[i+1];
                i++;
            } 
            else if ("-search".equals(args[i])) {
                search = args[i+1];
                i++;
            }
            else if ("-index".equals(args[i])) {
                indexPath = args[i+1];
                i++;
            }
            else if ("-topN".equals(args[i])) {
                topN = Integer.parseInt(args[i+1]);
                i++;
            }
        }
        System.out.println("search top " + topN + " results");
        if (testPath==null) {
            System.err.println("Usage: " + usage);
            System.exit(1);
        }
        

        Path testFile=Paths.get(testPath);
        if (!Files.isReadable(testFile)) {
            System.out.println("testFile '" +testFile.toAbsolutePath()+ "' does not exist or is not readable, please check the path");
            System.exit(1);
        }
        
        
        Date start = new Date();
        System.out.println("Loading Index directory '" + indexPath + "'...");      
        System.out.println("Loading Index Complete");
         try{
            //read test file
            ArrayList<String> testList = new ArrayList<>();
            FileReader testFr = new FileReader(testPath);
        		BufferedReader testBf = new BufferedReader(testFr);
        		String str=null;
        		while ((str = testBf.readLine()) != null) {
        			testList.add(str);
        		}
        		testFr.close();
        		testBf.close();
            System.out.println("test lines " + testList.size());
            
            //create result file
            File file = new File(testFile+".retrive."+topN);
            FileWriter fw = new FileWriter(file);
            IndexSearcher searcher = createSearcher(indexPath);
            for(int i=0; i<testList.size(); ++i){
                TopDocs foundDocs=null;
                if("source".equals(search)){
                    foundDocs = searchBySource(testList.get(i), searcher, topN);
                }
                else{
                    foundDocs = searchByTarget(testList.get(i), searcher, topN);
                } 
                if(i%100==0){
                    System.out.println("have proceeded " + i + " lines.");
                    System.out.println("Total Results: " + foundDocs.totalHits); 
                }
                fw.write("[APPEND] ");            
                if(foundDocs.scoreDocs.length==0){
                    fw.write("[SRC] [BLANK] [TGT] [BLANK] [SEP]\n");
                }
                else{
                    int count=0;
                    int flag=0;
                    for (ScoreDoc sd : foundDocs.scoreDocs) 
                    {         
                        Document d = searcher.doc(sd.doc);
                        if(d.get("source").equals(testList.get(i)) && search.equals("source")){ 
                            count += 1;
                            //System.out.println(d.get("source"));
                            //System.out.println("Skipping first result!");
                            continue;
                        }
                        if(d.get("target").equals(testList.get(i)) && search.equals("target")){ 
                            count += 1;
                            //System.out.println(d.get("target"));
                            //System.out.println("Skipping first result!");
                            continue;
                        }
                        //System.out.println(String.format(d.get("source")));
                        //System.out.println(String.format(d.get("target")));
                        if(count < foundDocs.scoreDocs.length - 1){
                            fw.write("[SRC] "+d.get("source")+" [TGT] "+d.get("target")+" [SEP] ");
                        }
                        else{
                            fw.write("[SRC] "+d.get("source")+" [TGT] "+d.get("target")+" [SEP]\n");
                            flag=1;
                            //System.out.println("Enter");
                        }
                        count += 1;
                    }
                    if(flag==0){
                        fw.write("[SRC] [BLANK] [TGT] [BLANK] [SEP]\n");
                    }
                }
                fw.flush();
            }            
            fw.close();
            Date end = new Date();
            System.out.println(end.getTime() - start.getTime() + " total milliseconds");
        } catch (IOException e) {
          System.out.println(" caught a " + e.getClass() + "\n with message: " + e.getMessage());
        }
        
         
    }
     
    private static TopDocs searchBySource(String source, IndexSearcher searcher, int topN) throws Exception
    {
        QueryParser qp = new QueryParser("source", new StandardAnalyzer());
        Query SourceQuery = qp.parse(QueryParser.escape(source).toLowerCase());
        TopDocs hits = searcher.search(SourceQuery, topN);
        return hits;
    }
    
    private static TopDocs searchByTarget(String target, IndexSearcher searcher, int topN) throws Exception
    {
        QueryParser qp = new QueryParser("target", new StandardAnalyzer());
        Query TargetQuery = qp.parse(QueryParser.escape(target).toLowerCase());
        TopDocs hits = searcher.search(TargetQuery, topN);
        return hits;
    }
  
    private static IndexSearcher createSearcher(String indexPath) throws IOException {
        Directory dir = FSDirectory.open(Paths.get(indexPath));
        IndexReader reader = DirectoryReader.open(dir);
        IndexSearcher searcher = new IndexSearcher(reader);
        return searcher;
    }
}