<template>
    <div class="container">
        <canvas ref="inference" style="width:100%;height:100%; position:fixed; left:0; top:0; z-index:-1"></canvas>
        <br>
        <div>
            <select v-model="selectedActivationFunctionName" @change="typeChange">
                <option v-for="item in activationFunctionNames" :value="item">
                    {{item}}
                </option>
            </select>
        </div>
        <br>

        <div>Number of layers: {{numLayers}}
        </div>
        <input @change="cppn.setNumLayers(numLayers);" type="range" step="1" min="0 " max="3" v-model="numLayers" id="layers-slider " />

        <div>z1 time coefficient</div>
        <input @change="cppn.setZ1Scale(convertZScale(z1Scale))" type="range" step="1" min=1 max=100 v-model="z1Scale"/>

        <div>z2 time coefficient</div>
        <input @change="cppn.setZ2Scale(convertZScale(z2Scale))" type="range" step="1" min="1" max=100 v-model="z2Scale"/>

        <input type="button" id="random">randomize</input>
        <br><br>
        <a href="http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/ " target="_blank ">What is a CPPN?</a>
    </div>
</template>
<script>
    import { ActivationFunction, CPPN } from './assets/cppn.ts';
    export default {
        data() {
            return {
                selectedActivationFunctionName: 'sin',
                activationFunctionNames: ['tanh', 'sin', 'relu'],
                CPPN: undefined,
                z1Scale: 1,
                z2Scale: 1,
                numLayers: 1,
                CANVAS_UPSCALE_FACTOR: 3,
                MAT_WIDTH: 30,
                WEIGHTS_STDEV: 0.6

            }
        },
        mounted() {
            this.init()
        },
        methods: {
            typeChange() {
                this.cppn.setActivationFunction(this.selectedActivationFunctionName)
            },
            convertZScale(z) {
                return (103 - z);
            },
            init() {
                this.inferenceCanvas = this.$refs.inference
                this.cppn = new CPPN(this.inferenceCanvas)

                // this.inferenceCanvas.style.width =
                //     `${this.inferenceCanvas.width * this.CANVAS_UPSCALE_FACTOR}px`
                // this.inferenceCanvas.style.height =
                //     `${this.inferenceCanvas.height * this.CANVAS_UPSCALE_FACTOR}px`;
                this.cppn.setActivationFunction(this.selectedActivationFunctionName)

                // const layersCountElement =
                //     document.querySelector('#layers-count') as HTMLDivElement;
                // layersSlider.addEventListener('immediate-value-changed', (event) => {
                // });
                this.cppn.setNumLayers(this.numLayers)

                // const z1Slider = document.querySelector('#z1-slider') as HTMLInputElement;
                // z1Slider.addEventListener('immediate-value-changed', (event) => {
                //     // tslint:disable-next-line:no-any
                //     this.z1Scale = parseInt((event as any).target.immediateValue, 10);
                //     this.;
                // });
                this.cppn.setZ1Scale(this.convertZScale(this.z1Scale))

                // const z2Slider = document.querySelector('#z2-slider') as HTMLInputElement;
                // z2Slider.addEventListener('immediate-value-changed', (event) => {
                //     // tslint:disable-next-line:no-any
                //     this.z2Scale = parseInt((event as any).target.immediateValue, 10);
                //     this.cppn.setZ2Scale(convertZScale(this.z2Scale));
                // });
                this.cppn.setZ2Scale(this.convertZScale(this.z2Scale))

                // const randomizeButton = document.querySelector('#random') as HTMLButtonElement;
                // randomizeButton.addEventListener('click', () => {
                //     this.cppn.generateWeights(MAT_WIDTH, WEIGHTS_STDEV);
                // });

                this.cppn.generateWeights(this.MAT_WIDTH, this.WEIGHTS_STDEV);
                this.cppn.start()
                console.log(this.cppn)
            }
        }
    }
</script>