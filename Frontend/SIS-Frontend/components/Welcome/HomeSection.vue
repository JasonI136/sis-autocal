<template>
  <section id="hero">
    <v-parallax dark src="">
      <v-row align="center" justify="center">
        <v-col cols="10">
          <v-row align="center" justify="center">
            <v-col cols="12" md="6" xl="8">
              <v-img max-width="30%" src="\img\autocalLogoTitle.png"></v-img>
              <h1 class="font-weight-light">
                Stop wasting time organising your week <br />
                Our Automated Calendar uses bleeding edge <u><b>AI</b></u> and <u><b>Machine Learning</b></u> <br />
                to make sure that you get the most of your day!
              </h1>


              <v-spacer></v-spacer>
              <!-- TODO: Figure out how tf to change this button to redirect to login page -->
              <!-- TODO 2: Change the navigation colour from orange to the AutoCal colours -->
              <div class="video d-flex align-center py-4">
                <v-btn to="/auth" class="playBut">
                  <!-- EVAN: Commented Out due to errors moved to v-btn -->
                  <!-- <a @click="$vuetify.go('/auth')"  class="playBut"> -->
                  <svg
                    version="1.1"
                    xmlns="http://www.w3.org/2000/svg"
                    xmlns:xlink="http://www.w3.org/1999/xlink"
                    xmlns:a="http://ns.adobe.com/AdobeSVGViewerExtensions/3.0/"
                    x="0px"
                    y="0px"
                    width="60px"
                    height="60px"
                    viewBox="0 0 213.7 213.7"
                    enable-background="new 0 0 213.7 213.7"
                    xml:space="preserve"
                  >
                    <polygon
                      class="triangle"
                      id="XMLID_18_"
                      fill="none"
                      stroke-width="7"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-miterlimit="10"
                      points="73.5,62.5 148.5,105.8 73.5,149.1 "
                    />

                    <circle
                      class="circle"
                      id="XMLID_17_"
                      fill="none"
                      stroke-width="7"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-miterlimit="10"
                      cx="106.8"
                      cy="106.8"
                      r="103.3"
                    />
                  </svg>
                <!-- </a> -->
                </v-btn>
                <p class="subheading ml-2 mb-0">Try out the calendar!</p>
              </div>
            </v-col>
            <v-col cols="12" md="6" xl="4" class="hidden-sm-and-down"> </v-col>
          </v-row>
        </v-col>
      </v-row>

    </v-parallax>
    <v-container fluid id="features" class="mt-2">
      <v-row align="center" justify="center">
        <v-col cols="10">
          <v-row align="center" justify="space-around">
            <!-- <v-col cols="12" class="text-center">
              <h1 class="font-weight-light display-2">Title</h1>
              <h1 class="font-weight-light">
                Lorem ipsum dolor sit amet consectetur adipisicing elit.
              </h1>
            </v-col> -->
            <v-col
              cols="12"
              sm="4"
              class="text-center"
              v-for="(feature, i) in features"
              :key="i"
            >
              <v-hover v-slot:default="{ hover }">
                <v-card
                  class="card"
                  shaped
                  :elevation="hover ? 10 : 4"
                  :class="{ up: hover }"
                >
                  <v-img
                    :src="feature.img"
                    max-width="100px"
                    class="d-block ml-auto mr-auto"
                    :class="{ 'zoom-efect': hover }"
                  ></v-img>
                  <h1 class="font-weight-regular">{{ feature.title }}</h1>
                  <h4 class="font-weight-regular subtitle-1">
                    {{ feature.text }}
                  </h4>
                </v-card>
              </v-hover>
            </v-col>
          </v-row>
        </v-col>
      </v-row>
    </v-container>
    <v-dialog v-model="dialog" max-width="640px">
      <v-card>

      </v-card>
    </v-dialog>
    <div class="svg-border-waves">
      <div class="svg-border-waves text-white">
        <v-img src="https://landingpagecalangoweb.web.app/img/borderWaves.b5381f92.svg" />
      </div>
      <img src="~@/static/img/wave2.svg" />
    </div>
  </section>
</template>

<script>
export default {
  data() {
    return {
      dialog: false,
      videoId: "i8IvvHJssWE",
      features: [
        {
          img: require("@/static/img/icon2.png"),
          title: "AI Predictions",
          text: "Using artificial intelligence, the calendar will be able to fill in your activities and schedule your day with a press of a button",
        },
        {
          img: require("@/static/img/icon3.png"),
          title: "Changes for you",
          text: "The Calendar will learn what you like and dont like. Scheduling your day will get easier and easier the more you use the app",
        },
        {
          img: require("@/static/img/icon1.png"),
          title: "Privacy in Mind",
          text: "While the AutoCal will know your schedule, no one else will! Your schedule is your data and we wont steal it like Facebook or Meta",
        },
      ],
    };
  },
  watch: {

  },
  methods: {

    login(event){
      this.$router.push({ name: 'calendar' })
    },


  },
};
</script>

<style lang="scss">
.circle {
  stroke: white;
  stroke-dasharray: 650;
  stroke-dashoffset: 650;
  -webkit-transition: all 0.5s ease-in-out;
  opacity: 0.3;
}

.playBut {
  /*  border: 1px solid red;*/
  // display: inline-block;
  -webkit-transition: all 0.5s ease;

  .triangle {
    -webkit-transition: all 0.7s ease-in-out;
    stroke-dasharray: 240;
    stroke-dashoffset: 480;
    stroke: white;
    transform: translateY(0);
  }

  &:hover {
    .triangle {
      stroke-dashoffset: 0;
      opacity: 1;
      stroke: white;
      animation: nudge 0.7s ease-in-out;

      @keyframes nudge {
        0% {
          transform: translateX(0);
        }
        30% {
          transform: translateX(-5px);
        }
        50% {
          transform: translateX(5px);
        }
        70% {
          transform: translateX(-2px);
        }
        100% {
          transform: translateX(0);
        }
      }
    }

    .circle {
      stroke-dashoffset: 0;
      opacity: 1;
    }
  }
}
</style>

<style>
.btn-play {
  transition: 0.2s;
}

.svg-border-waves .v-image {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 3rem;
  width: 100%;
  overflow: hidden;
}

#hero {
  z-index: 0;
}
.svg-border-waves img {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  margin-bottom: -2px;
  z-index: -1;
}

.card {
  min-height: 300px;
  padding: 10px;
  transition: 0.5s ease-out;
}

.card .v-image {
  margin-bottom: 15px;
  transition: 0.75s;
}

.card h1 {
  margin-bottom: 10px;
}

.zoom-efect {
  transform: scale(1.1);
}

.up {
  transform: translateY(-20px);
  transition: 0.5s ease-out;
}
</style>

<style>
section {
  position: relative;
}
</style>
