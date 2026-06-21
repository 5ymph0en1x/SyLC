static float4 gl_Position;
static float2 v_texCoord;
static float2 texCoord;
static float2 position;

struct SPIRV_Cross_Input
{
    float2 position : TEXCOORD0;
    float2 texCoord : TEXCOORD1;
};

struct SPIRV_Cross_Output
{
    float2 v_texCoord : TEXCOORD0;
    float4 gl_Position : SV_Position;
};

void vert_main()
{
    v_texCoord = texCoord;
    gl_Position = float4(position, 0.0f, 1.0f);
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    texCoord = stage_input.texCoord;
    position = stage_input.position;
    vert_main();
    SPIRV_Cross_Output stage_output;
    stage_output.gl_Position = gl_Position;
    stage_output.v_texCoord = v_texCoord;
    return stage_output;
}
