//package com.javatpoint;  
public class Reader {

  void read(NeuralNetwork nn) {
    try {
      // use buffered reader to read line by line
      BufferedReader reader = createReader("m1.txt");
      int i;
      String line = null;
      for (i=0; i<nn.inputs; i++) {
        for (int j=0; j<nn.hidden; j++){
          line = reader.readLine();
          nn.m1.data[i][j] = float(line);
        }
      }
      reader = createReader("m2.txt");
      for (i=0; i<nn.hidden; i++) {
        for (int j=0; j<nn.outputs; j++){
          line = reader.readLine();
          nn.m2.data[i][j] = float(line);
        }
      }
    } 
    catch (IOException e) {
      println("Exception:" + e.toString());
    }
  }

  void write(NeuralNetwork nn) {
    PrintWriter output = createWriter("m1.txt"); 
    for (int i=0; i<nn.inputs; i++) {
      for (int j=0; j<nn.hidden; j++)
        output.println(nn.m1.data[i][j]);
    }
    output.flush();
    output.close();
    output = createWriter("m2.txt"); 
    for (int i=0; i<nn.hidden; i++) {
      for (int j=0; j<nn.outputs; j++)
        output.println(nn.m2.data[i][j]);
    }
    output.flush();
    output.close();
  }
}
