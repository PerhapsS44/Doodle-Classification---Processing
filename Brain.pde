public class NeuralNetwork{
  public int inputs;
  public int hidden;
  public int outputs;
  
  float lr = 0.005;
  
  Matrix m1;
  Matrix m2; 
  Matrix bh;
  Matrix bo;
  NeuralNetwork(int a, int b, int c){
    inputs = a;
    hidden = b;
    outputs = c;
    
    m1 = new Matrix(inputs,hidden);
    m2 = new Matrix(hidden,outputs);
    
    bh = new Matrix(1,hidden);
    bo = new Matrix(1,outputs);
  }
  
  float[] estimate(float[] in){
    
    float[] out = new float[outputs];
    
    Matrix in_m = new Matrix(1,inputs);
    Matrix out_m = new Matrix(1,outputs);
    in_m.data[0] = in;
    
    Matrix hidden_m = product(in_m,m1);
    hidden_m.sum(bh);
    for (int i=0;i<hidden_m.r;i++){
      for (int j=0;j<hidden_m.c;j++){
        hidden_m.data[i][j] = sigmoid(hidden_m.data[i][j]);
      }
    }
    
    out_m = product(hidden_m,m2);
    out_m.sum(bo);
    for (int i=0;i<out_m.r;i++){
      for (int j=0;j<out_m.c;j++){
        out_m.data[i][j] = sigmoid(out_m.data[i][j]);
      }
    }
    out = out_m.data[0];
    
    //println(out[0]);
    
    return out;
  }
  
  void evolve(float[] in, float[] tar){
    
    float[] out = new float[outputs];
    
    Matrix in_m = new Matrix(1,inputs);
    Matrix out_m = new Matrix(1,outputs);
    
    Matrix err_bh = new Matrix(1,hidden);
    Matrix err_bo = new Matrix(1,outputs);
    in_m.data[0] = in;
    
    Matrix hidden_m = product(in_m,m1);
    
    hidden_m.sum(bh);
    for (int i=0;i<hidden_m.r;i++){
      for (int j=0;j<hidden_m.c;j++){
        hidden_m.data[i][j] = sigmoid(hidden_m.data[i][j]);
      }
    }
    
    out_m = product(hidden_m,m2);
    out_m.sum(bo);
    for (int i=0;i<out_m.r;i++){
      for (int j=0;j<out_m.c;j++){
        out_m.data[i][j] = sigmoid(out_m.data[i][j]);
      }
    }
    out = out_m.data[0];
    
    Matrix err = new Matrix(1,outputs);
    float[] errors = new float[outputs];

    for (int i=0;i<outputs;i++){
      errors[i] = tar[i]-out[i];

      err.data[0][i] = errors[i];
    }

    //float[] h_errors = product(err,m2.transpose()).data[0];
    Matrix h_err = product(m2,err.transpose());
    float[] h_errors = new float[h_err.r];
    for (int i=0;i<h_err.r;i++){
      h_errors[i] = h_err.data[i][0];
    }
    Matrix error_ho = new Matrix(m2.r,m2.c);
    
    //println(error_ho.r+"  "+error_ho.c);
    
    for (int i=0;i<error_ho.r;i++){
      for (int j=0;j<error_ho.c;j++){
        error_ho.data[i][j] = lr * errors[j] * dsigmoid(out[j]) * hidden_m.data[0][i];
        //print(error_ho.data[i][j] + " ");
      }
      //println();
    }
    
    
    Matrix error_ih = new Matrix(m1.r,m1.c);
    
    for (int i=0;i<error_ih.r;i++){
      for (int j=0;j<error_ih.c;j++){
        error_ih.data[i][j] *= lr;
        error_ih.data[i][j] *= h_errors[j];
        error_ih.data[i][j] *= dsigmoid(hidden_m.data[0][j]);
        error_ih.data[i][j] *= in[i];
        //print(h_errors[j] + " ");
        }
      //println();
    }
    
    for (int i=0;i<err_bo.r;i++){
      for (int j=0;j<err_bo.c;j++){
        err_bo.data[i][j] = lr * errors[j] * dsigmoid(out[j]);
      }
    }
    
    for (int i=0;i<err_bh.r;i++){
      for (int j=0;j<err_bh.c;j++){
        err_bh.data[i][j] *= lr;
        err_bh.data[i][j] *= h_errors[j];
        err_bh.data[i][j] *= dsigmoid(hidden_m.data[0][j]);
      }
    }
    
    for (int i=0;i<err_bh.r;i++){
      for (int j=0;j<err_bh.c;j++){
        bh.data[i][j]+=err_bh.data[i][j];
      }
    }
    
    for (int i=0;i<err_bo.r;i++){
      for (int j=0;j<err_bo.c;j++){
        bo.data[i][j] +=err_bo.data[i][j];
      }
    }
    
    
    for (int i=0;i<m1.r;i++){
      for (int j=0;j<m1.c;j++){
        m1.data[i][j] += error_ih.data[i][j];
        
      }
    }
    
    
    for (int i=0;i<m2.r;i++){
      for (int j=0;j<m2.c;j++){
        m2.data[i][j] += error_ho.data[i][j];
      }
    }
    
    
  }
  
}

public class Matrix{
  
  int r;
  int c;
  float[][] data;
  Matrix(int _r, int _c){
    r = _r;
    c = _c;
    data = new float[r][c];
    
    for (int i=0;i<r;i++){
      for (int j=0;j<c;j++){
        data[i][j] = random(-1,1);
      }
    }
  }
  Matrix transpose(){
    Matrix out = new Matrix(this.c,this.r);
    for (int i=0;i<this.r;i++){
      for (int j=0;j<this.c;j++){
        out.data[j][i] = this.data[i][j];
      }
    }
    return out;
  }
  
  void sum(Matrix m){
   for (int i=0;i<this.r;i++){
      for (int j=0;j<this.c;j++){
        this.data[i][j] += m.data[i][j];
      }
    }
  }
  
  void printData(){
    for (int i=0;i<this.r;i++){
      for (int j=0;j<this.c;j++){
        print(this.data[i][j]+" ");
      }
      print("\n");
    }
  }
  
}

Matrix product(Matrix m1, Matrix m2){
  Matrix rezult = new Matrix(m1.r,m2.c);
  
  for (int i=0;i<rezult.r;i++){
    for (int k=0;k<rezult.c;k++){
      rezult.data[i][k] = 0;
    }
  }
  
  for (int i=0;i<m1.r;i++){
    for (int k=0;k<m2.c;k++){
      for (int j=0;j<m1.c;j++){
        rezult.data[i][k]+=m1.data[i][j]*m2.data[j][k];
      }
    }
  }
  return rezult;
}

float sigmoid(float x){
  return 1/(1+exp(-x));
}
float dsigmoid(float y){
  return y*(1-y);
}
