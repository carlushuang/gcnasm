#include <stdio.h>
#include <math.h>

typedef unsigned int   uint32;

enum //F8 formats
{
    BF8_FMT,
    FP8_FMT
};

enum DATA_TYPE
{
    FP16,
    BF16,
    FP8,
    FP32,
    TYPE_NUM
};
double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    //srand((unsigned)time(NULL));
    if ( phase == 0 ) 
    {
        do 
        {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } 
        while(S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    } 
    else
       X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;
    return X;
}
template<typename T>
static void fmha_dumpMatrixInHex(T *buffer,  FILE *file, int m, int n, DATA_TYPE in_type, int transpose = 0)
{
    if(transpose == 0)
    {
        for(int i = 0; i < m ; i++)
        {
            fprintf(file, "R[%04d]: ", i);
            for(int j = 0; j < n ; j++)
            {
                T value;
                value = buffer[i * n + j];
                
                if((in_type == FP16) || (in_type == BF16))
                   fprintf(file, "0x%04x ", (*((uint32*)&value) & 0xffff));
                else if (in_type == FP8) //FP8
                   fprintf(file, "0x%02x ", (*((uint32*)&value) & 0xff));
                else if (in_type == FP32) //FP32
                   fprintf(file, "0x%08x ", (*((uint32*)&value) & 0xffffffff));
            }
            fprintf(file, "\n");
        }
    }
    else
    {
        for(int j = 0; j < n ; j++)
        {
            fprintf(file, "R[%04d]: ", j);
            for(int i = 0; i < m ; i++)
            {
                T value;
                value = buffer[i * n + j];
                
                if((in_type == FP16) || (in_type == BF16))
                   fprintf(file, "0x%04x ", (*((uint32*)&value) & 0xffff));
                else if (in_type == FP8) //FP8
                   fprintf(file, "0x%02x ", (*((uint32*)&value) & 0xff));
                else if (in_type == FP32) //FP32
                   fprintf(file, "0x%08x ", (*((uint32*)&value) & 0xffffffff));
            }
            fprintf(file, "\n");
        }
    }
}

template<typename T>
static void fmha_dump_batch_inHex(T *buffer, const char *fileName, int batch, int head_num, int m, int n, DATA_TYPE in_type, int transpose = 0)
{
    FILE *file = fopen(fileName, "w+t");

    for(int b = 0; b < batch; b++)
    {
        for(int h = 0; h < head_num; h++)
        {
            fprintf(file, "++++Batch[%04d]---head[%04d]++++: \n", b, h);
            fmha_dumpMatrixInHex(buffer+b*head_num*m*n+h*m*n, file, m, n, in_type, transpose);
        }
    }

    fclose(file);
}

template<typename T>
static void fmha_batch_init(T *buffer, int batch, int head_num, int seq_len, int head_dim, DATA_TYPE in_type, int init_pattern = 0, int fp_format = FP8_FMT, bool f8_bias = false)
{
    for(int b = 0; b < batch; b++)
    {
        for(int h = 0; h < head_num; h++)
        {
            for(int s = 0; s < seq_len; s++)
            {
                for(int d = 0; d < head_dim; d++)
                {
                    int offset = b * head_num * seq_len * head_dim  + h * seq_len * head_dim + s * head_dim + d;
 
                    float temp_var;
                    switch (init_pattern)
                    {
                        case 0:
                            temp_var = (float)gaussrand();
                            break;
                        //case 1:
                        //    temp_var = cos(offset);
                        //    break;
                        //case 2:
                        //    temp_var = sin(offset);
                        //    break;
                        //case 3:
                        //    temp_var = cos(offset) + sin(offset);
                        //    break;
                        case 10:
                            temp_var = 0.25;
                            break;
                        case 11:
                            temp_var = 0.01*d;
                            break;
                        case 12:
                            temp_var = 0.01*s;
                            break;
                        default:
                            temp_var = 0;
                            break;
                    }

                    switch(in_type)
                    {
                        case FP16:
                             //buffer[offset] = (uint16)FP32toFP16(FloatMapToInt(temp_var));
                             buffer[offset] = __float2half_rn(temp_var);
                             break;
                        case BF16:
                             //buffer[offset] = (uint16)FP32toBFP16(FloatMapToInt(temp_var));
                             buffer[offset] = __float2half_rn(temp_var);
                             break;
                        //case FP8:
                        //     buffer[offset] = f32_to_fp8(FloatMapToInt(temp_var), 127, fp_format, f8_bias, true, false, 0);
                        //     break;
                        case FP32:
                             buffer[offset] = temp_var;
                             break;
                        default:
                             break;
                    }
                }//head_dim
            }//seq_len
        }//head_num
    }//batch
}

template<typename T>
void fmha_batch_cvt(T *output, float *a, int batch, int head_num, int seq_len, int head_dim, DATA_TYPE type, int fp_format_des = FP8_FMT, bool f8_bias_des = false)
{
    for(int b = 0; b < batch; b++)
    {
        for(int h = 0; h < head_num; h++)
        {
            for(int s = 0; s < seq_len; s++)
            {
                for(int d = 0; d < head_dim; d++)
                {
                    int offset = b * head_num * seq_len * head_dim  + h * seq_len * head_dim + s * head_dim + d;
                    output[offset] = __float2half_rn(a[offset]);
                }
            }
        }
    }
}
void fmha_bwd_dQ_redc(float *dq, int batch, int head_num, int seq_len, int head_dim, int split)
{
    for(int b = 0; b < batch; b++)
    {
        for(int h = 0; h < head_num; h++)
        {
            for(int s = 0; s < seq_len; s++)
            {
                for(int d = 0; d < head_dim; d++)
                {
                   float sum = 0.0;

                   int o_offs =  b * head_num * seq_len * head_dim  + h * seq_len * head_dim + s * head_dim + d;
                   int i_offs =  b * head_num * seq_len * head_dim * split + h * seq_len * head_dim * split + s * head_dim + d;
                   for(int i = 0; i < split; i++)
                   {
                      sum += dq[i_offs];
                      i_offs += head_dim*seq_len;
                   }
                   
                   dq[o_offs] = sum;
                }
            }
        }
    }
}
