<template>
  <div>
    <canvas id="myChart"></canvas>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';


export default {

  data: () => ({
    chart: null,
    chartData: {
      type: "doughnut",
      data: {
        labels: [],
        datasets: [
          {
            label: 'Data One',
            backgroundColor: [
              'rgba(255, 99, 132, 1)',
              'rgba(54, 162, 235, 1)',
              'rgba(255, 206, 86, 1)',
              'rgba(75, 192, 192, 1)',
              'rgba(153, 102, 255, 1)',
              'rgba(255, 159, 64, 1)'
            ],
            borderColor: 'rgb(255, 255, 255)',
            data: []
          }
        ]
      },
      options: {
        responsive: true,
      }
    },
  }),

  props: {
    events: Array
  },
  computed: {
    calendar: {
      get() {
        return this.$store.state.calendar;
      },
    }
  },
  watch: {
    calendar() {
      this.getEventsPerWeek();
    },
    events: {
      handler() {
        this.getEventsPerWeek();
      },
      deep: true
    }


  },
  mounted() {
    const ctx = document.getElementById('myChart');
    this.chart = new Chart(ctx, this.chartData);
  },
  created() {
    this.getEventsPerWeek();
  },
  methods: {
    recalculateTable() {
      if (this.chart) {
        this.chart.update();
      }
    },

    getEventsPerWeek() {
      const startWeek = this.$moment(this.calendar, "YYYY-MM-DD").startOf('isoWeek').valueOf()
      const endWeek = this.$moment(this.calendar, "YYYY-MM-DD").endOf('isoWeek').valueOf()
      const eventsCurrentWeek = this.events.filter(x => x.start >= startWeek && x.start <= endWeek)

      let result = eventsCurrentWeek.reduce(function (r, a) {
        r[a.category] = r[a.category] || [];
        r[a.category].push(a);
        return r;
      }, Object.create(null));


      var ids = Object.keys(result);
      this.chartData.data.labels = ids

      let eventData = [];
      let eventColors = []

      // loop all elements within ids
      // loop all the events within elements
      ids.forEach(element => {
        var count = 0;
        for(let i = 0; i < result[element].length; i++){
          count += result[element][i].end - result[element][i].start
        }
        let hours = (count / (1000 * 60 * 60)).toFixed(1);
        eventData.push(hours);
        eventColors.push(result[element][0].color)
      });
      console.log(result)
      this.chartData.data.datasets[0].data = eventData;
      // EVAN: Array of Respective Category Color. Comment out for now due to color naming scheme
      // this.chartData.data.datasets[0].backgroundColor = eventColors;
      console.log(eventColors)
      this.recalculateTable();
      return ids
    }
  }
}
</script>
