import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Date;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;


import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;


public class IndexFiles {
  
    private IndexFiles() {}


    public static void main(String[] args) {
        String usage = "java org.apache.lucene.demo.IndexFiles"
                     + " [-index INDEX_PATH] [-source SOURCE_PATH] [-target TARGET_PATH]\n\n"
                     + "This indexes the documents in DOCS_PATH, creating a Lucene index"
                     + "in INDEX_PATH that can be searched with SearchFiles";
        String indexPath = "index";
        String sourcePath=null;
        String targetPath=null;
        String docsPath = null;
        boolean merge = false;
        for(int i=0;i<args.length;i++) {
            if ("-source".equals(args[i])) {
                sourcePath = args[i+1];
                i++;
            } 
            else if ("-target".equals(args[i])) {
                targetPath = args[i+1];
                i++;
            }
            else if ("-index".equals(args[i])) {
                indexPath = args[i+1];
                i++;
            }
            else if ("-merge".equals(args[i])) {
                merge = true;
            }
        }
        if (sourcePath==null || targetPath==null) {
            System.err.println("Usage: " + usage);
            System.exit(1);
        }
    
        Path sourceFile=Paths.get(sourcePath);
        Path targetFile=Paths.get(targetPath);
        if (!Files.isReadable(sourceFile)) {
            System.out.println("sourceFile '" +sourceFile.toAbsolutePath()+ "' does not exist or is not readable, please check the path");
            System.exit(1);
        }
        if (!Files.isReadable(targetFile)) {
            System.out.println("targetFile '" +targetFile.toAbsolutePath()+ "' does not exist or is not readable, please check the path");
            System.exit(1);
        }
        
        Date start = new Date();
        System.out.println("Indexing to directory '" + indexPath + "'...");
        
        try{
            //read source file
            ArrayList<String> sourceList = new ArrayList<>();
            FileReader sourceFr = new FileReader(sourcePath);
        		BufferedReader sourceBf = new BufferedReader(sourceFr);
        		String str=null;
        		while ((str = sourceBf.readLine()) != null) {
        			sourceList.add(str);
        		}
        		sourceFr.close();
        		sourceBf.close();
           System.out.println("source lines "+sourceList.size());
            //read target file
            ArrayList<String> targetList = new ArrayList<>();
            FileReader targetFr = new FileReader(targetPath);
        		BufferedReader targetBf = new BufferedReader(targetFr);
        		while ((str = targetBf.readLine()) != null) {
        			targetList.add(str);
        		}
        		targetFr.close();
        		targetBf.close();
            System.out.println("target lines "+targetList.size());
            if(sourceList.size()!= targetList.size()){
                System.out.println("source and target lines are not equal!");
                System.exit(1);
            }
            List<Document> documents = new ArrayList<>();
            for (int i = 0; i < sourceList.size(); i++){
          			String sourceSent=sourceList.get(i);
                String targetSent=targetList.get(i);
                Document document = createDocument(i, sourceSent, targetSent);
                documents.add(document);
        		}
           IndexWriter writer = createWriter(indexPath);
           writer.deleteAll();
           writer.addDocuments(documents);
           writer.commit();    
           if(merge){  
             writer.forceMerge(1);
           }
           writer.close();
      
           Date end = new Date();
           System.out.println(end.getTime() - start.getTime() + " total milliseconds");
        } catch (IOException e) {
          System.out.println(" caught a " + e.getClass() + "\n with message: " + e.getMessage());
        }
        
    
      
    }
    
    private static IndexWriter createWriter(String indexPath)
    {
        try{
            FSDirectory dir = FSDirectory.open(Paths.get(indexPath));
            IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
            config.setRAMBufferSizeMB(256.0);
            IndexWriter writer = new IndexWriter(dir, config);       
            return writer;   
        } catch (IOException e) {
            System.out.println(" caught a " + e.getClass() + "\n with message: " + e.getMessage());
            return null;
        }
        
    }
  
    private static Document createDocument(Integer id, String sourceSent, String targetSent) 
    {
        Document document = new Document();
        document.add(new StringField("id", id.toString() , Field.Store.YES));       
        document.add(new TextField("source", sourceSent , Field.Store.YES));
        document.add(new TextField("target", targetSent , Field.Store.YES));
        return document;
    }
    


  
}